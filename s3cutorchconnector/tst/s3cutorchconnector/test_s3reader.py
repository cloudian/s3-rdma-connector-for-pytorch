#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import uuid
import io
import os

import pytest

from s3cutorchconnector.s3client import S3RdmaClient
from s3cutorchconnector.s3reader import S3Reader
from s3cutorchconnector._types import S3BucketKey
from s3cutorchconnector.reader_constructor import S3ReaderConstructor


def _reader_tests(reader, start_off, contents):
    length = len(contents)

    # test seek before reading
    reader.seek(start_off, io.SEEK_SET)
    assert reader.read() == contents

    # test seek after reading using readinto()
    reader.seek(start_off, io.SEEK_SET)
    data = bytearray(length)
    assert reader.readinto(data) == length
    assert data == contents

    # A series of reads through entire object
    step = (length // 5) + 1  # Ensure we straddle chunk boundaries
    reader.seek(start_off, io.SEEK_SET)
    for off in range(0, length, step):
        to_read = min(step, length-off)
        assert reader.read(to_read) == contents[off:off+to_read]


def test_s3reader():
    chunk_size = 10 << 20
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = str(chunk_size >> 20)
    client = S3RdmaClient()
    bucket_name = f"test-s3reader-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        object_sizes = [
            20,  # small object
            chunk_size,  # 1 chunk
            3 * chunk_size,  # 3 chunks
            3 * chunk_size + 1,  # 3 full chunks + 1 byte
            3 * chunk_size - 1,  # 2 full chunks + one nearly full chunk
        ]

        for with_objinfo in [True, False]:
            for length in object_sizes:
                contents = io.BytesIO()
                for i in range(length):
                    contents.write(bytes([i % 256]))
                contents = contents.getvalue()

                client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
                bkey = S3BucketKey(bucket_name, "obj")
                if with_objinfo:
                    bkey.ensure_object_info(client)

                # Full read
                print('full', length, with_objinfo)
                reader = S3Reader(client, bkey)
                _reader_tests(reader, 0, contents)

                # Read first 5 bytes
                print('first', length, with_objinfo)
                reader = S3Reader(client, bkey, read_range=(0, 5))
                _reader_tests(reader, 0, contents[:5])

                # Read last 5 bytes
                print('last', length, with_objinfo)
                reader = S3Reader(client, bkey, read_range=(length - 5, length))
                _reader_tests(reader, length - 5, contents[-5:])

                # Read 10 bytes from the middle
                midpoint = length // 2
                print('middle', length, with_objinfo, midpoint)
                reader = S3Reader(client, bkey, read_range=(midpoint - 5, midpoint + 5))
                _reader_tests(reader, midpoint - 5, contents[midpoint - 5:midpoint + 5])

    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# Confirm S3ReaderConstructor.sequential() and .range_based() both read back identical data.
def test_s3reader_constructor():
    chunk_size = 10 << 20
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = str(chunk_size >> 20)
    client = S3RdmaClient()
    bucket_name = f"test-s3reader-constructor-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        length = 3 * chunk_size + 1  # 3 full chunks + 1 byte, so both readers span multiple chunks
        contents = bytes(i % 256 for i in range(length))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        for constructor in [S3ReaderConstructor.sequential(),
                            S3ReaderConstructor.range_based(),
                            S3ReaderConstructor.range_based(max_buffers=2)]:
            reader = constructor(client, bkey)
            _reader_tests(reader, 0, contents)
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# Exercise the paged reader's prefetch window, including a non-zero read_range combined with
# max_buffers (the offset math this previously got wrong for the paged/lazy cache-miss path).
def test_s3reader_prefetch():
    chunk_size = 10 << 20
    os.environ['S3RDMA_BUFFER_POOL_BUFSIZE_MIB'] = str(chunk_size >> 20)
    client = S3RdmaClient()
    bucket_name = f"test-s3reader-prefetch-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        length = 5 * chunk_size + 1  # 5 full chunks + 1 byte
        contents = bytes(i % 256 for i in range(length))
        client.s3_client.put_object(Bucket=bucket_name, Key="obj", Body=contents)
        bkey = S3BucketKey(bucket_name, "obj")

        # Full object, various prefetch depths (0 disables prefetch, and depth is clamped to
        # max_buffers - 1 so an oversized depth shouldn't break anything)
        for max_buffers, prefetch_depth in [(2, 0), (2, 1), (4, 1), (4, 10)]:
            reader = S3Reader(client, bkey, max_buffers=max_buffers, prefetch_depth=prefetch_depth)
            _reader_tests(reader, 0, contents)

        # Non-zero read_range spanning multiple chunks, combined with paged/prefetching mode
        mid_start = chunk_size - 5
        mid_end = mid_start + 3 * chunk_size + 10
        reader = S3Reader(client, bkey, read_range=(mid_start, mid_end), max_buffers=2, prefetch_depth=1)
        _reader_tests(reader, mid_start, contents[mid_start:mid_end])
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)


# A zero-length read_range (e.g. (5, 5)) must short-circuit without ever touching S3 - there's
# no valid Range header for an empty range, and there's nothing to fetch anyway. Uses a
# nonexistent bucket/key: if the reader ever actually made a request, this would fail loudly.
def test_s3reader_zero_length_range():
    client = S3RdmaClient()
    bkey = S3BucketKey("this-bucket-does-not-exist-at-all", "no-such-key")

    for read_range in [(5, 5), (0, 0)]:
        reader = S3Reader(client, bkey, read_range=read_range)
        assert reader.read() == b''
        assert reader.readinto(bytearray(0)) == 0
        reader.close()

    # seek() still works within the (empty) range, using absolute-position semantics
    reader = S3Reader(client, bkey, read_range=(5, 5))
    assert reader.seek(5, io.SEEK_SET) == 0
    assert reader.seek(0, io.SEEK_END) == 0
    with pytest.raises(ValueError):
        reader.seek(0, io.SEEK_SET)
    reader.close()


if __name__ == "__main__":
    test_s3reader()
    test_s3reader_constructor()
    test_s3reader_prefetch()
    test_s3reader_zero_length_range()
