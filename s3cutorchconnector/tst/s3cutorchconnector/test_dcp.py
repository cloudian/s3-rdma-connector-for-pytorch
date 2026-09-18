#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import functools
import uuid

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.planner import SavePlan

from s3cutorchconnector.s3client import S3RdmaClient
from s3cutorchconnector.s3reader import S3Reader
from s3cutorchconnector.dcp import S3FileSystem, S3StorageReader, S3StorageWriter, BinaryPrefixStrategy
from s3cutorchconnector.dcp.s3_file_system import StorageMetadata
from s3cutorchconnector.reader_constructor import S3ReaderConstructor, DcpOptimizedConstructor


def test_s3_filesystem():
    client = S3RdmaClient()
    bucket_name = f"test-fs-{uuid.uuid4()}"
    key = "test-key.txt"

    s3_uri = "s3://" + bucket_name + "/" + key
    client.s3_client.create_bucket(Bucket=bucket_name)
    fs = S3FileSystem()

    try:
        # Below chunk size, test PutObject
        contents = b'data contents 2'

        with fs.create_stream(s3_uri, "wb") as writer:
            writer.write(contents)

        b = bytearray(len(contents))
        with fs.create_stream(s3_uri, "rb") as reader:
            reader.readinto(b)

        assert b == contents

        fs.rename(s3_uri, s3_uri + ".renamed")
        fs.rm_file(s3_uri + ".renamed")

    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key=key)
        client.s3_client.delete_object(Bucket=bucket_name, Key=key + ".renamed")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# Confirm create_stream honours an explicit range_based reader_constructor.
def test_s3_filesystem_range_based_reader():
    client = S3RdmaClient()
    bucket_name = f"test-fs-range-{uuid.uuid4()}"
    key = "test-key.txt"

    s3_uri = "s3://" + bucket_name + "/" + key
    client.s3_client.create_bucket(Bucket=bucket_name)
    fs = S3FileSystem(reader_constructor=S3ReaderConstructor.range_based(max_buffers=2))

    try:
        contents = b'data contents for range based reader'

        with fs.create_stream(s3_uri, "wb") as writer:
            writer.write(contents)

        b = bytearray(len(contents))
        with fs.create_stream(s3_uri, "rb") as reader:
            reader.readinto(b)

        assert b == contents
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key=key)
        client.s3_client.delete_bucket(Bucket=bucket_name)


# S3StorageReader should default to the lazy range_based reader (so DCP loads don't eagerly
# download whole checkpoint shard files), while still allowing callers to override it.
# pylint: disable=protected-access
def test_s3_storage_reader_default_reader_constructor():
    # sequential() (and default()) build a partial(S3Reader, max_buffers=None); range_based()
    # builds a distinct closure, so a partial bound to S3Reader with max_buffers=None uniquely
    # identifies the eager/sequential reader.
    def is_sequential(constructor) -> bool:
        return (isinstance(constructor, functools.partial)
                and constructor.func is S3Reader
                and constructor.keywords.get("max_buffers") is None)

    reader = S3StorageReader(None, "s3://some-bucket/some-path")
    assert not is_sequential(reader._reader_constructor)

    overridden = S3StorageReader(
        None, "s3://some-bucket/some-path",
        reader_constructor=S3ReaderConstructor.sequential(),
    )
    assert is_sequential(overridden._reader_constructor)


# Full DCP save/load round trip using the opt-in dcp_optimized() reader, confirming it produces
# correct results end to end (coalescing, buffer packing, and the .metadata fallback all get
# exercised via the real save/load flow) and that item ranges actually get injected.
# pylint: disable=protected-access
def test_dcp_save_and_load_with_dcp_optimized_reader():
    client = S3RdmaClient()
    bucket_name = f"test-dcp-optimized-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)
    s3_uri = f"s3://{bucket_name}/ckpt"

    try:
        state_dict = {f"tensor_{i}": torch.arange(2000 * (i + 1), dtype=torch.float32) for i in range(8)}

        writer = S3StorageWriter(None, s3_uri)
        dcp.save(state_dict, storage_writer=writer)

        reader_constructor = S3ReaderConstructor.dcp_optimized(max_buffers=2)
        reader = S3StorageReader(None, s3_uri, reader_constructor=reader_constructor)
        assert isinstance(reader._reader_constructor, DcpOptimizedConstructor)

        loaded = {k: torch.zeros_like(v) for k, v in state_dict.items()}
        dcp.load(loaded, storage_reader=reader)

        for k in state_dict:
            assert torch.equal(state_dict[k], loaded[k])

        assert reader_constructor._item_ranges_by_file, "no item ranges were recorded"
    finally:
        objs = client.s3_client.list_objects_v2(Bucket=bucket_name, Prefix="ckpt")
        for obj in objs.get("Contents", []):
            client.s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        client.s3_client.delete_bucket(Bucket=bucket_name)


# A real DCP save/load round trip with a non-default prefix_strategy, confirming both that the
# resulting S3 keys actually carry the expected prefix, and that a plain S3StorageReader (no
# prefix_strategy needed for reads - the prefix is baked into the saved metadata) still loads it.
def test_dcp_save_with_prefix_strategy():
    client = S3RdmaClient()
    bucket_name = f"test-dcp-prefix-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)
    s3_uri = f"s3://{bucket_name}/ckpt"

    try:
        state_dict = {"tensor_0": torch.arange(1000, dtype=torch.float32)}

        # prefix_count is pinned explicitly (rather than left to default to the distributed world
        # size) so the expected "/0000/" prefix below doesn't depend on whether some other test in
        # this process happens to have initialized torch.distributed with a different world size.
        writer = S3StorageWriter(
            None, s3_uri,
            prefix_strategy=BinaryPrefixStrategy(min_prefix_length=4, prefix_count=1))
        dcp.save(state_dict, storage_writer=writer)

        objs = client.s3_client.list_objects_v2(Bucket=bucket_name, Prefix="ckpt")
        keys = [obj["Key"] for obj in objs.get("Contents", [])]
        assert any("/0000/" in key for key in keys), f"no key carries the expected prefix: {keys}"

        reader = S3StorageReader(None, s3_uri)
        loaded = {k: torch.zeros_like(v) for k, v in state_dict.items()}
        dcp.load(loaded, storage_reader=reader)
        assert torch.equal(state_dict["tensor_0"], loaded["tensor_0"])
    finally:
        objs = client.s3_client.list_objects_v2(Bucket=bucket_name, Prefix="ckpt")
        for obj in objs.get("Contents", []):
            client.s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        client.s3_client.delete_bucket(Bucket=bucket_name)


# prepare_global_plan must not clobber storage_data FileSystemWriter's own prepare_local_plan
# already set (the use_collectives=False / per-rank path) - only fill it in when it's still
# unset, exactly like FileSystemWriter's own prepare_global_plan does. Otherwise every rank's
# plan would be rewritten using its position in the (possibly single-element, per-rank) `plans`
# list rather than its real rank, colliding every rank onto the same shard names.
def test_prepare_global_plan_preserves_existing_storage_data():
    writer = S3StorageWriter(
        None, "s3://some-bucket/some-path",
        prefix_strategy=BinaryPrefixStrategy(prefix_count=1, min_prefix_length=1))

    already_set = object()
    plan_with_storage_data = SavePlan(items=[], storage_data=already_set)
    plan_without_storage_data = SavePlan(items=[])

    result = writer.prepare_global_plan([plan_with_storage_data, plan_without_storage_data])

    assert result[0].storage_data is already_set
    assert isinstance(result[1].storage_data, StorageMetadata)
