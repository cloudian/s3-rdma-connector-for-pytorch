#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

# pylint: disable=protected-access
# These tests deliberately inspect internal state (_reads_by_buffer, pending_buffers, etc.) to
# verify buffer lifecycle/prefetch behavior that isn't otherwise observable from outside.

import os
import uuid

import pytest

from s3cutorchconnector.s3client import S3RdmaClient
from s3cutorchconnector._types import S3BucketKey
from s3cutorchconnector.reader_constructor import DcpOptimizedConstructor
from s3cutorchconnector.dcp_optimized_reader import (
    DcpPlannedS3Reader,
    ItemRange,
    PlannedRead,
    _coalesce_item_ranges,
    _plan_buffer_reads,
)


def test_coalesce_merges_gap_below_threshold():
    ranges = [ItemRange(0, 10), ItemRange(15, 20)]
    assert _coalesce_item_ranges(ranges, max_gap_size=10) == [(0, 20)]


def test_coalesce_keeps_gap_above_threshold_separate():
    ranges = [ItemRange(0, 10), ItemRange(50, 60)]
    assert _coalesce_item_ranges(ranges, max_gap_size=10) == [(0, 10), (50, 60)]


def test_coalesce_gap_exactly_at_threshold_merges():
    # gap (10) == max_gap_size (10): boundary case, should still merge ("<=", not "<")
    ranges = [ItemRange(0, 10), ItemRange(20, 30)]
    assert _coalesce_item_ranges(ranges, max_gap_size=10) == [(0, 30)]


def test_coalesce_gap_one_over_threshold_stays_separate():
    ranges = [ItemRange(0, 10), ItemRange(21, 30)]
    assert _coalesce_item_ranges(ranges, max_gap_size=10) == [(0, 10), (21, 30)]


def test_coalesce_filters_zero_length_ranges():
    ranges = [ItemRange(0, 10), ItemRange(10, 10), ItemRange(20, 30)]
    assert _coalesce_item_ranges(ranges, max_gap_size=0) == [(0, 10), (20, 30)]


def test_coalesce_all_zero_length_raises():
    with pytest.raises(ValueError):
        _coalesce_item_ranges([ItemRange(5, 5)], max_gap_size=10)


def test_coalesce_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="Overlapping"):
        _coalesce_item_ranges([ItemRange(0, 10), ItemRange(5, 15)], max_gap_size=10)


def test_coalesce_rejects_unsorted_ranges():
    with pytest.raises(ValueError, match="Unsorted"):
        _coalesce_item_ranges([ItemRange(10, 20), ItemRange(0, 5)], max_gap_size=10)


def test_plan_span_exactly_fills_one_buffer():
    plan = _plan_buffer_reads([(0, 100)], chunk_size=100)
    assert plan == [PlannedRead(start=0, length=100, buffer_idx=0, buffer_offset=0)]


def test_plan_span_larger_than_buffer_is_split():
    plan = _plan_buffer_reads([(0, 250)], chunk_size=100)
    assert plan == [
        PlannedRead(start=0, length=100, buffer_idx=0, buffer_offset=0),
        PlannedRead(start=100, length=100, buffer_idx=1, buffer_offset=0),
        PlannedRead(start=200, length=50, buffer_idx=2, buffer_offset=0),
    ]


def test_plan_packs_multiple_small_spans_into_one_buffer():
    plan = _plan_buffer_reads([(0, 10), (20, 30)], chunk_size=100)
    assert plan == [
        PlannedRead(start=0, length=10, buffer_idx=0, buffer_offset=0),
        PlannedRead(start=20, length=10, buffer_idx=0, buffer_offset=10),
    ]


def test_plan_splits_a_span_that_straddles_a_buffer_boundary():
    # First span leaves 5 bytes free in buffer 0. The second span (length 10) fills that
    # leftover space first, then spills the remaining 5 bytes into a new buffer - packing is
    # purely greedy, it never holds a span back to avoid splitting it.
    plan = _plan_buffer_reads([(0, 95), (100, 110)], chunk_size=100)
    assert plan == [
        PlannedRead(start=0, length=95, buffer_idx=0, buffer_offset=0),
        PlannedRead(start=100, length=5, buffer_idx=0, buffer_offset=95),
        PlannedRead(start=105, length=5, buffer_idx=1, buffer_offset=0),
    ]


# Live-server tests exercising DcpPlannedS3Reader's actual RDMA reads: coalescing, buffer
# packing/sharing, a buffer-boundary-spanning item, and the foreground fallback.
def test_dcp_planned_reader_reads_match_and_fall_back_outside_the_plan():
    chunk_size = 1 << 20  # 1 MiB
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = "1"
    client = S3RdmaClient()
    bucket_name = f"test-dcp-reader-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        length = 3 * chunk_size
        contents = bytes(i % 256 for i in range(length))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        item_ranges = [
            ItemRange(100, 200),
            ItemRange(250, 300),  # gap 50, merges with the previous item into one buffer read
            ItemRange(chunk_size - 100, chunk_size + 100),  # shares buffer 0's leftover space
            ItemRange(2 * chunk_size + 500, 2 * chunk_size + 600),  # far away, separate span
        ]

        reader = DcpPlannedS3Reader(
            client, bkey, item_ranges, max_gap_size=1000, max_buffers=2, prefetch_depth=1)
        try:
            for r in item_ranges:
                reader.seek(r.start)
                assert reader.read(r.end - r.start) == contents[r.start:r.end]

            for r in item_ranges:
                reader.seek(r.start)
                buf = bytearray(r.end - r.start)
                assert reader.readinto(buf) == len(buf)
                assert bytes(buf) == contents[r.start:r.end]

            # Foreground fallback: bytes not covered by any item range.
            reader.seek(1000)
            assert reader.read(50) == contents[1000:1050]

            reader.seek(1000)
            buf = bytearray(50)
            reader.readinto(buf)
            assert bytes(buf) == contents[1000:1050]
        finally:
            reader.close()
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


def test_dcp_planned_reader_item_spanning_multiple_buffers():
    chunk_size = 1 << 20  # 1 MiB
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = "1"
    client = S3RdmaClient()
    bucket_name = f"test-dcp-multibuf-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        length = 5 * chunk_size
        contents = bytes(i % 256 for i in range(length))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        # 2.5 buffers' worth of data in one item - must split across 3 buffers.
        big_item = ItemRange(0, int(2.5 * chunk_size))
        reader = DcpPlannedS3Reader(client, bkey, [big_item], max_buffers=2, prefetch_depth=1)
        assert len({pr.buffer_idx for pr in reader.plan}) == 3

        try:
            reader.seek(big_item.start)
            assert reader.read(big_item.end - big_item.start) == contents[:big_item.end]
        finally:
            reader.close()
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# A DCP load can legitimately contain only empty items (e.g. an empty tensor); the constructor
# fallback must not build a DcpPlannedS3Reader in that case, since its coalescer requires at
# least one non-empty range.
def test_all_zero_length_ranges_do_not_construct_a_planned_reader():
    client = S3RdmaClient()
    bkey = S3BucketKey("this-bucket-does-not-exist-at-all", "no-such-key")
    constructor = DcpOptimizedConstructor()
    constructor._item_ranges_by_file[f"s3://{bkey.bucket}/{bkey.key}"] = [ItemRange(5, 5), ItemRange(10, 10)]

    reader = constructor(client, bkey)
    assert not isinstance(reader, DcpPlannedS3Reader)


def test_max_buffers_zero_is_rejected():
    client = S3RdmaClient()
    bkey = S3BucketKey("this-bucket-does-not-exist-at-all", "no-such-key")
    with pytest.raises(ValueError):
        DcpPlannedS3Reader(client, bkey, [ItemRange(0, 10)], max_buffers=0)


def test_read_clamps_to_remaining_object_size():
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = "1"
    client = S3RdmaClient()
    bucket_name = f"test-dcp-clamp-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        contents = bytes(range(100))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        reader = DcpPlannedS3Reader(client, bkey, [ItemRange(0, 100)])
        try:
            reader.seek(90)
            # Ask for far more than remains; must clamp instead of overrunning the object.
            data = reader.read(1000)
            assert data == contents[90:100]

            reader.seek(90)
            buf = bytearray(1000)
            n = reader.readinto(buf)
            assert n == 10
            assert bytes(buf[:10]) == contents[90:100]
        finally:
            reader.close()
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# Sequential access must keep the prefetch window sliding forward: once a buffer is fully
# consumed it should be released immediately, freeing budget for the next prefetch, rather than
# lingering in the cache and starving _schedule_prefetch.
def test_prefetch_window_keeps_sliding_past_the_first_transition():
    chunk_size = 1 << 20  # 1 MiB
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = "1"
    client = S3RdmaClient()
    bucket_name = f"test-dcp-window-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        length = 5 * chunk_size
        contents = bytes(i % 256 for i in range(length))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        big_item = ItemRange(0, length)
        reader = DcpPlannedS3Reader(client, bkey, [big_item], max_buffers=2, prefetch_depth=1)
        try:
            reader.seek(0)
            # Cross from buffer 0 into buffer 1 - this is the transition where the window
            # previously stalled.
            reader.read(chunk_size + 1)

            assert 0 not in reader.lru_cache, "stale buffer 0 should have been released"
            assert 2 in reader.pending_buffers or 2 in reader.lru_cache, \
                "buffer 2 should already be scheduled/fetched, not stalled"
        finally:
            reader.close()
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


def test_fetch_buffer_releases_its_buffer_on_a_mid_fetch_exception(monkeypatch):
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = "1"
    client = S3RdmaClient()
    bucket_name = f"test-dcp-leak-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        contents = bytes(500020)
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        # Two ranges far enough apart to stay as separate spans (not coalesced), but both small
        # enough to land in the same buffer - so fetching buffer 0 takes two separate
        # get_object_direct calls.
        item_ranges = [ItemRange(0, 10), ItemRange(500000, 500010)]
        reader = DcpPlannedS3Reader(client, bkey, item_ranges, max_gap_size=100, max_buffers=2)
        assert len(reader._reads_by_buffer[0]) == 2

        pool = client.s3_client.buffer_pool
        available_before = pool.available()

        real_get_object_direct = client.s3_client.get_object_direct
        call_count = {"n": 0}

        def failing_get_object_direct(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated transient failure")
            return real_get_object_direct(*args, **kwargs)

        monkeypatch.setattr(client.s3_client, "get_object_direct", failing_get_object_direct)

        with pytest.raises(RuntimeError):
            reader._fetch_buffer(0)

        assert pool.available() == available_before, "buffer was leaked on the mid-fetch exception"
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)
