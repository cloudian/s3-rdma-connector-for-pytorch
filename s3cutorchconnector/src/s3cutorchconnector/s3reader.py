#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import typing
from concurrent import futures
import pylru

from .s3client import S3RdmaClient, create_read_buffer
from ._types import S3BucketKey


# pylint: disable=too-many-instance-attributes
class S3Reader(io.BufferedIOBase):
    # pylint: disable=too-many-arguments
    def __init__(self, client: S3RdmaClient, meta: S3BucketKey, *,
                 read_range: typing.Union[tuple[int, int], None] = None,
                 max_buffers: typing.Optional[int] = None,
                 prefetch_depth: int = 0):
        self.client = client
        self.chunk_size = client.s3_client.buffer_pool.buffer_size()
        self.meta = meta
        # Cursor is always relative to the beginning of the read_range (or object if no read_range)
        self.cursor = 0
        self.chunks = None
        self.thread_pool = None
        self.lru_cache = None
        self.match_obj = {}
        self.begin = read_range[0] if read_range else 0
        self.end = read_range[1] if read_range else None
        self.len = self.end - self.begin if read_range else None
        self.fetched_chunk_zero = False

        # Prefetch state, used only in paged (max_buffers) mode. Clamp so we never prefetch
        # further ahead than the LRU cache can hold, which would just evict work before it's used.
        self.prefetch_depth = max(0, min(prefetch_depth, max_buffers - 1)) if max_buffers else 0
        self.prefetch_pool = None
        self.pending_chunks: typing.Dict[int, futures.Future] = {}

        # Configure LRU cache if we're limiting the maximum number of RDMA buffers to use
        if max_buffers:
            if max_buffers < 1:
                raise ValueError("max_buffers must be at least 1")

            # Return evicted chunks to the pool for reuse
            def return_evicted_chunk(_k, chunk):
                self.client.s3_client.buffer_pool.put_buffer(chunk)

            self.lru_cache = pylru.lrucache(max_buffers, return_evicted_chunk)

    @property
    def bucket(self):
        return self.meta.bucket

    @property
    def key(self):
        return self.meta.key

    @property
    def object_info(self):
        return self.meta.object_info

    def prefetch(self):
        self._ensure_reader()

    def detach(self):
        raise io.UnsupportedOperation

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self):
        if self.prefetch_pool:
            self.prefetch_pool.shutdown()
            # Drain any in-flight prefetches so their buffers aren't leaked
            for chunk_future in self.pending_chunks.values():
                self.client.s3_client.buffer_pool.put_buffer(chunk_future.result())
            self.pending_chunks.clear()
            self.prefetch_pool = None

        if self.thread_pool:
            self.thread_pool.shutdown()
            for chunk in self.chunks:
                self.client.s3_client.buffer_pool.put_buffer(chunk.result())
            self.thread_pool = None
        elif self.chunks:
            self.client.s3_client.buffer_pool.put_buffer(self.chunks)
        elif self.lru_cache is not None:
            for chunk in self.lru_cache.values():
                self.client.s3_client.buffer_pool.put_buffer(chunk)
            self.lru_cache.clear()
        self.chunks = None
        self.cursor = 0

    def read(self, size=-1, /) -> bytes:
        self._ensure_reader()
        if size < 0 or size is None:
            to_read = self.len - self.cursor
        else:
            to_read = min(size, self.len - self.cursor)

        buf = create_read_buffer(to_read)
        self._readinto(buf, to_read)
        return buf

    def readinto(self, b, /) -> int:
        if memoryview(b).readonly:  # pylint: disable=using-constant-test
            raise ValueError("buffer is read only")

        self._ensure_reader()
        to_read = min(len(b), self.len - self.cursor)
        self._readinto(b, to_read)
        return to_read

    def seek(self, offset, whence=io.SEEK_SET) -> int:
        self._ensure_reader()
        if whence == io.SEEK_SET:
            offset -= self.begin
        elif whence == io.SEEK_CUR:
            offset += self.cursor
        elif whence == io.SEEK_END:
            offset += self.len
        else:
            raise ValueError("Seek must be passed SEEK_CUR, SEEK_SET, or SEEK_END")

        if offset < 0:
            raise ValueError("Seek before beginning of range")
        offset = min(offset, self.len)
        self.cursor = offset
        return offset

    def _readinto(self, dst, read_len):
        """ Reads read_len bytes into the dst buffer """

        end = self.cursor + read_len
        if read_len == 0:
            self.cursor = end
            return

        if self.len > self.chunk_size:
            begin_idx = self.cursor // self.chunk_size
            end_idx = (end + self.chunk_size - 1) // self.chunk_size

            dst_offset = 0
            chunk_off = self.cursor % self.chunk_size
            for i in range(begin_idx, end_idx):
                chunk = self._get_chunk(i)
                chunk_len = min(read_len, self.chunk_size - chunk_off)
                chunk.slice(chunk_off, chunk_len).copy_to_buffer(dst, dst_offset)
                dst_offset += chunk_len
                chunk_off = 0
                read_len -= chunk_len
        else:
            # All data in single chunk
            self.chunks.slice(self.cursor, read_len).copy_to_buffer(dst, 0)

        self.cursor = end

    # pylint: disable=too-many-branches
    def _ensure_reader(self):
        if self.fetched_chunk_zero:
            return

        # Zero-length read_range (e.g. (5, 5)): nothing to fetch, and there's no valid Range
        # header for an empty range anyway, so skip the network call entirely.
        if self.len == 0:
            self.fetched_chunk_zero = True
            return

        # Read the first chunk to determine total length and etag for additional reads
        first_chunk = self.client.s3_client.buffer_pool.get_buffer()
        range_arg = {'Range': f'bytes={self.begin}-{self.end - 1}'} if self.end is not None else {}
        rsp = self.client.s3_client.get_object_direct(first_chunk.data_ptr(), Bucket=self.meta.bucket, Key=self.meta.key, **range_arg)

        # Check how much data has been transferred. RdmaBytesTransferred is now just a byte
        # count (no longer "<transferred>/<total>"). If we don't already know the range length
        # (self.end), the object's total size comes from the Content-Range response header
        # instead (its denominator is always the full object size, not the requested range's
        # length, so it's only used to fill in self.end/self.len - Content-Range is absent only
        # for a zero-byte object). Checked with "is None", not truthiness: self.end == 0 is a
        # valid, already-known state (an explicit zero-length read_range), not "unknown".
        transferred = rsp['RdmaBytesTransferred']
        if self.end is None:
            self.end = int(rsp['ContentRange'].split("/")[-1]) if transferred else 0
            self.len = self.end

        self.fetched_chunk_zero = True
        if transferred == self.len:
            # We've already read the data
            self.chunks = first_chunk
        elif self.lru_cache is not None:
            # Ensure additional reads use the same object (or fail if it has changed)
            self.match_obj = {'VersionId': rsp['VersionId']} if 'VersionId' in rsp else {'IfMatch': rsp['ETag']}
            self.lru_cache[0] = first_chunk
            if self.prefetch_depth:
                self.prefetch_pool = futures.ThreadPoolExecutor(max_workers=self.client.max_concurrent_reads)
                self._schedule_prefetch(0)
        else:
            self.thread_pool = futures.ThreadPoolExecutor(max_workers=self.client.max_concurrent_reads)
            self.chunks = []
            # Create a dummy future to indicate the first chunk read is complete
            fut = futures.Future()
            fut.set_result(first_chunk)
            self.chunks.append(fut)
            off = self.chunk_size
            # Ensure additional reads use the same object (or fail if it has changed)
            self.match_obj = {'VersionId': rsp['VersionId']} if 'VersionId' in rsp else {'IfMatch': rsp['ETag']}

            while off < self.end:
                end = min(self.end, off + self.chunk_size)
                self.chunks.append(self._async_read_chunk(off, end))
                off = end

    def _get_chunk(self, idx):
        # If we're not in paged mode, all chunks are scheduled to be read
        if self.lru_cache is None:
            # Wait until our chunk is ready
            return self.chunks[idx].result()

        # We're in paged mode, check if we have the chunk already
        if idx in self.lru_cache:
            chunk = self.lru_cache[idx]
        elif idx in self.pending_chunks:
            # A prefetch for this chunk is already in flight; wait for it
            chunk = self.pending_chunks.pop(idx).result()
            self.lru_cache[idx] = chunk
        else:
            # Cache miss, no prefetch in flight either - fetch synchronously
            begin, end = self._chunk_bounds(idx)
            chunk = self._read_chunk(begin, end)
            self.lru_cache[idx] = chunk

        if self.prefetch_pool:
            self._schedule_prefetch(idx)

        return chunk

    def _chunk_bounds(self, idx: int) -> typing.Tuple[int, int]:
        """ Byte range of chunk idx, relative to self.begin (as _read_chunk expects) """
        begin = idx * self.chunk_size
        end = min(begin + self.chunk_size, self.len)
        return begin, end

    def _schedule_prefetch(self, from_idx: int):
        """ Kick off background fetches for upcoming chunks, up to prefetch_depth ahead.

        Stays within the lru_cache's overall capacity (cached + in-flight chunks combined),
        so a single reader never holds more than max_buffers RDMA buffers at once.
        """
        last_idx = (self.len - 1) // self.chunk_size
        budget = self.lru_cache.size() - len(self.lru_cache) - len(self.pending_chunks)
        for idx in range(from_idx + 1, min(from_idx + 1 + self.prefetch_depth, last_idx + 1)):
            if budget <= 0:
                break
            if idx in self.lru_cache or idx in self.pending_chunks:
                continue
            begin, end = self._chunk_bounds(idx)
            self.pending_chunks[idx] = self.prefetch_pool.submit(self._read_chunk, begin, end)
            budget -= 1

    def _async_read_chunk(self, begin, end):
        return self.thread_pool.submit(lambda: self._read_chunk(begin, end))

    def _read_chunk(self, begin, end):
        # Adjust offsets to be within the initial read_range
        begin += self.begin
        end += self.begin
        chunk = self.client.s3_client.buffer_pool.get_buffer()
        self.client.s3_client.get_object_direct(chunk.data_ptr(),
                                                Bucket=self.meta.bucket, Key=self.meta.key,
                                                Range=f'bytes={begin}-{end-1}', **self.match_obj)
        return chunk
