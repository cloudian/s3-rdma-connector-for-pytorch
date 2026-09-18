#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import bisect
import io
import typing
from concurrent import futures
from dataclasses import dataclass

import pylru

from .s3client import S3RdmaClient
from .s3reader import S3Reader
from ._types import S3BucketKey


# Default max gap (bytes) between item ranges such that they still get fetched together in one
# buffer-packing pass - half of the default 10 MB RDMA buffer size.
DEFAULT_MAX_GAP_SIZE = 5 * 1024 * 1024


@dataclass
class ItemRange:
    """Byte range for a single DCP ReadItem (tensor). Inclusive start, exclusive end,
    absolute offsets within the object."""
    start: int
    end: int


@dataclass
class PlannedRead:
    """One get_object_direct call: fetch [start, start+length) of the object into
    buffer_idx at buffer_offset."""
    start: int
    length: int
    buffer_idx: int
    buffer_offset: int


# Coalesce a sorted, non-overlapping list of ItemRanges into (start, end) spans, merging
# consecutive ranges whose gap is <= max_gap_size so they can share a buffer/fetch.
def _coalesce_item_ranges(
        item_ranges: typing.List[ItemRange],
        max_gap_size: typing.Union[int, float]) -> typing.List[typing.Tuple[int, int]]:
    ranges = [r for r in item_ranges if r.end != r.start]
    if not ranges:
        raise ValueError("No non-empty ranges to read (all ranges were length 0)")

    first = ranges[0]
    if first.start < 0 or first.end < first.start:
        raise ValueError(f"Invalid range: {first.start}-{first.end}")

    spans = []
    group_start, group_end = first.start, first.end
    for r in ranges[1:]:
        if r.start < 0 or r.end < r.start:
            raise ValueError(f"Invalid range: {r.start}-{r.end}")
        if r.start < group_end:
            if r.start < group_start:
                raise ValueError(f"Unsorted ranges: {group_start}-{group_end} and {r.start}-{r.end}")
            raise ValueError(f"Overlapping ranges: {group_start}-{group_end} and {r.start}-{r.end}")

        if r.start - group_end <= max_gap_size:
            group_end = r.end
        else:
            spans.append((group_start, group_end))
            group_start, group_end = r.start, r.end

    spans.append((group_start, group_end))
    return spans


# Pack coalesced spans into a sequence of chunk_size-capacity buffers, splitting a span across
# buffers if it doesn't fit, and packing multiple spans into one buffer's leftover space if it does.
def _plan_buffer_reads(spans: typing.List[typing.Tuple[int, int]], chunk_size: int) -> typing.List[PlannedRead]:
    plan = []
    buffer_idx = 0
    bytes_used = 0

    for span_start, span_end in spans:
        offset = span_start
        remaining = span_end - span_start
        while remaining > 0:
            capacity = chunk_size - bytes_used
            if capacity == 0:
                buffer_idx += 1
                bytes_used = 0
                capacity = chunk_size

            take = min(remaining, capacity)
            plan.append(PlannedRead(start=offset, length=take, buffer_idx=buffer_idx, buffer_offset=bytes_used))
            bytes_used += take
            offset += take
            remaining -= take

    return plan


# pylint: disable=too-many-instance-attributes
class DcpPlannedS3Reader(io.BufferedIOBase):
    """Reads exactly the byte ranges a DCP load plan needs, packing them into as few RDMA
    buffers as possible. Falls back to a plain foreground S3Reader for any access outside the
    planned ranges (shouldn't happen if the caller sticks to the plan)."""

    # pylint: disable=too-many-arguments
    def __init__(self, client: S3RdmaClient, meta: S3BucketKey, item_ranges: typing.List[ItemRange], *,
                 read_range: typing.Union[typing.Tuple[int, int], None] = None,
                 max_gap_size: typing.Union[int, float] = DEFAULT_MAX_GAP_SIZE,
                 max_buffers: typing.Optional[int] = None,
                 prefetch_depth: int = 1):
        if read_range is not None:
            raise NotImplementedError("DcpPlannedS3Reader does not support read_range")

        self.client = client
        self.meta = meta
        self.chunk_size = client.s3_client.buffer_pool.buffer_size()
        self.cursor = 0
        self.match_obj: typing.Dict[str, typing.Any] = {}

        spans = _coalesce_item_ranges(item_ranges, max_gap_size)
        self.plan = _plan_buffer_reads(spans, self.chunk_size)
        self._plan_starts = [pr.start for pr in self.plan]

        self._reads_by_buffer: typing.Dict[int, typing.List[PlannedRead]] = {}
        for pr in self.plan:
            self._reads_by_buffer.setdefault(pr.buffer_idx, []).append(pr)
        self._last_buffer_idx = self.plan[-1].buffer_idx

        if max_buffers is not None and max_buffers < 1:
            raise ValueError("max_buffers must be at least 1")
        max_buffers = max_buffers or client.max_concurrent_reads
        self.prefetch_depth = max(0, min(prefetch_depth, max_buffers - 1))

        def return_evicted_buffer(_k, buf):
            self.client.s3_client.buffer_pool.put_buffer(buf)

        self.lru_cache = pylru.lrucache(max_buffers, return_evicted_buffer)
        self.pending_buffers: typing.Dict[int, futures.Future] = {}
        self.prefetch_pool = None
        self._fetched_first = False

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
            for buf_future in self.pending_buffers.values():
                self.client.s3_client.buffer_pool.put_buffer(buf_future.result())
            self.pending_buffers.clear()
            self.prefetch_pool = None

        for buf in self.lru_cache.values():
            self.client.s3_client.buffer_pool.put_buffer(buf)
        self.lru_cache.clear()

    def read(self, size=-1, /) -> bytes:
        self._ensure_reader()
        remaining = self._object_size() - self.cursor
        to_read = remaining if size is None or size < 0 else min(size, remaining)
        buf = bytearray(to_read)
        self._readinto(buf, to_read)
        return bytes(buf)

    def readinto(self, b, /) -> int:
        if memoryview(b).readonly:  # pylint: disable=using-constant-test
            raise ValueError("buffer is read only")
        self._ensure_reader()
        to_read = min(len(b), self._object_size() - self.cursor)
        self._readinto(b, to_read)
        return to_read

    def seek(self, offset, whence=io.SEEK_SET) -> int:
        self._ensure_reader()
        if whence == io.SEEK_SET:
            pass
        elif whence == io.SEEK_CUR:
            offset += self.cursor
        elif whence == io.SEEK_END:
            offset += self._object_size()
        else:
            raise ValueError("Seek must be passed SEEK_CUR, SEEK_SET, or SEEK_END")

        if offset < 0:
            raise ValueError("Seek before beginning of object")
        self.cursor = offset
        return offset

    def _object_size(self) -> int:
        self.meta.ensure_object_info(self.client)
        return self.meta.object_info.size

    def _ensure_reader(self):
        if self._fetched_first:
            return
        self._fetched_first = True

        # Prime match_obj (ETag/VersionId) from the very first buffer, synchronously, before any
        # concurrent prefetching starts - avoids a race establishing it from multiple threads.
        self.lru_cache[0] = self._fetch_buffer(0)
        if self.prefetch_depth:
            self.prefetch_pool = futures.ThreadPoolExecutor(max_workers=self.client.max_concurrent_reads)
            self._schedule_prefetch(0)

    def _readinto(self, dst, read_len):
        """ Reads read_len bytes into dst, starting at self.cursor. Falls back to a foreground
        S3Reader read for any part of the request not covered by the plan. """
        dst_offset = 0
        while read_len > 0:
            pr, pr_idx = self._planned_read_at(self.cursor)
            if pr is None:
                # Not covered by the plan - foreground fallback for exactly what's needed here,
                # bounded by where plan coverage resumes (if it ever does).
                end = self.cursor + read_len
                next_start = self._plan_starts[pr_idx] if pr_idx < len(self.plan) else end
                fallback_len = min(read_len, max(1, next_start - self.cursor))
                self._foreground_read(dst, dst_offset, fallback_len)
                dst_offset += fallback_len
                self.cursor += fallback_len
                read_len -= fallback_len
                continue

            buf = self._get_buffer(pr.buffer_idx)
            offset_in_read = self.cursor - pr.start
            avail = pr.length - offset_in_read
            take = min(read_len, avail)
            buf.slice(pr.buffer_offset + offset_in_read, take).copy_to_buffer(dst, dst_offset)

            dst_offset += take
            self.cursor += take
            read_len -= take

    def _planned_read_at(self, pos: int) -> typing.Tuple[typing.Optional[PlannedRead], int]:
        """ Returns (the PlannedRead covering pos, its index), or (None, index of the next
        PlannedRead after pos) if pos isn't covered by the plan. """
        idx = bisect.bisect_right(self._plan_starts, pos) - 1
        if idx >= 0 and pos < self.plan[idx].start + self.plan[idx].length:
            return self.plan[idx], idx
        return None, idx + 1

    def _foreground_read(self, dst, dst_offset: int, length: int):
        reader = S3Reader(self.client, self.meta, read_range=(self.cursor, self.cursor + length))
        try:
            reader.readinto(memoryview(dst)[dst_offset:dst_offset + length])
        finally:
            reader.close()

    def _get_buffer(self, buffer_idx: int):
        # Access is strictly sequential (guaranteed by S3StorageReader.prepare_local_plan's
        # offset sort), so any cached buffer before this index is fully consumed and will never
        # be touched again. Free it now rather than waiting for the LRU cache's own eviction on
        # the next insert - otherwise stale-but-still-cached buffers eat into the budget
        # _schedule_prefetch uses to decide how far ahead it can prefetch, stalling the window.
        self._release_stale_buffers(buffer_idx)

        if buffer_idx in self.lru_cache:
            buf = self.lru_cache[buffer_idx]
        elif buffer_idx in self.pending_buffers:
            buf = self.pending_buffers.pop(buffer_idx).result()
            self.lru_cache[buffer_idx] = buf
        else:
            buf = self._fetch_buffer(buffer_idx)
            self.lru_cache[buffer_idx] = buf

        if self.prefetch_pool:
            self._schedule_prefetch(buffer_idx)
        return buf

    def _release_stale_buffers(self, up_to_idx: int):
        """ Evict and return to the pool any cached buffer strictly before up_to_idx """
        for idx in [i for i in self.lru_cache if i < up_to_idx]:
            self.client.s3_client.buffer_pool.put_buffer(self.lru_cache.pop(idx))

    def _schedule_prefetch(self, from_idx: int):
        budget = self.lru_cache.size() - len(self.lru_cache) - len(self.pending_buffers)
        idx = from_idx + 1
        scheduled = 0
        while budget > 0 and scheduled < self.prefetch_depth and idx <= self._last_buffer_idx:
            if idx in self._reads_by_buffer and idx not in self.lru_cache and idx not in self.pending_buffers:
                self.pending_buffers[idx] = self.prefetch_pool.submit(self._fetch_buffer, idx)
                budget -= 1
                scheduled += 1
            idx += 1

    def _fetch_buffer(self, buffer_idx: int):
        """ Fetches every PlannedRead targeting buffer_idx into one registered buffer. """
        buf = self.client.s3_client.buffer_pool.get_buffer()
        try:
            for pr in self._reads_by_buffer[buffer_idx]:
                dst_ptr = buf.data_ptr().slice(pr.buffer_offset, pr.length)
                rsp = self.client.s3_client.get_object_direct(
                    dst_ptr, Bucket=self.meta.bucket, Key=self.meta.key,
                    Range=f'bytes={pr.start}-{pr.start + pr.length - 1}', **self.match_obj)
                if not self.match_obj:
                    self.match_obj = {'VersionId': rsp['VersionId']} if 'VersionId' in rsp else {'IfMatch': rsp['ETag']}
        except Exception:
            self.client.s3_client.buffer_pool.put_buffer(buf)
            raise
        return buf
