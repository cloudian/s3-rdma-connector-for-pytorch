#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from functools import partial
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from .s3client import S3RdmaClient
from .s3reader import S3Reader
from ._types import S3BucketKey
from .dcp_optimized_reader import DcpPlannedS3Reader, ItemRange, DEFAULT_MAX_GAP_SIZE

if TYPE_CHECKING:
    from torch.distributed.checkpoint.planner import ReadItem
    from torch.distributed.checkpoint.metadata import MetadataIndex
    from torch.distributed.checkpoint.filesystem import _StorageInfo


# Callable used to build an S3Reader for a given client/object, mirroring S3Reader's own signature.
S3ReaderConstructorProtocol = Callable[..., S3Reader]


class S3ReaderConstructor:
    """Factory for S3Reader-constructing callables, selecting between our two reader strategies:
    eager whole-object prefetch ("sequential") and LRU-paged lazy per-chunk fetch ("range_based").
    """

    @staticmethod
    def sequential() -> S3ReaderConstructorProtocol:
        """Constructor for the eager, whole-object prefetching reader (S3Reader's current default)."""
        return partial(S3Reader, max_buffers=None)

    @staticmethod
    def range_based(max_buffers: Optional[int] = None, prefetch_depth: int = 1) -> S3ReaderConstructorProtocol:
        """Constructor for the LRU-paged lazy reader, which only fetches chunks that are actually read.

        Args:
            max_buffers: Maximum number of RDMA buffers to keep resident at once. Defaults to
                the client's max_concurrent_reads if not specified.
            prefetch_depth: How many chunks ahead of the current read position to fetch in the
                background, overlapping network latency with the caller's processing of the
                current chunk. Set to 0 to fetch strictly on demand. Clamped to max_buffers - 1
                by S3Reader, since a reader never holds more than max_buffers buffers at once.
        """
        if max_buffers is not None and max_buffers < 1:
            raise ValueError("max_buffers must be at least 1")

        def _construct(
            client: S3RdmaClient, meta: S3BucketKey, *,
            read_range: Optional[Tuple[int, int]] = None,
        ) -> S3Reader:
            return S3Reader(
                client, meta, read_range=read_range,
                max_buffers=max_buffers or client.max_concurrent_reads,
                prefetch_depth=prefetch_depth,
            )

        return _construct

    @staticmethod
    def default() -> S3ReaderConstructorProtocol:
        """Constructor used when no explicit reader_constructor is supplied."""
        return S3ReaderConstructor.sequential()

    @staticmethod
    def dcp_optimized(
            max_gap_size: Union[int, float] = DEFAULT_MAX_GAP_SIZE,
            max_buffers: Optional[int] = None,
            prefetch_depth: int = 1) -> "DcpOptimizedConstructor":
        """Constructor for a reader that fetches exactly the byte ranges a DCP load plan needs,
        packed into as few RDMA buffers as possible. Opt-in: pass to S3StorageReader's
        reader_constructor. See DcpOptimizedConstructor for details.
        """
        return DcpOptimizedConstructor(
            max_gap_size=max_gap_size, max_buffers=max_buffers, prefetch_depth=prefetch_depth)


class DcpOptimizedConstructor:
    """Constructor for DcpPlannedS3Reader instances, with byte ranges injected from a DCP load
    plan. Created via S3ReaderConstructor.dcp_optimized() and used by S3StorageReader.

    Usage flow:
        S3StorageReader(..., reader_constructor=S3ReaderConstructor.dcp_optimized())
            -> prepare_local_plan(plan) -> set_item_ranges_by_file(plan.items, storage_data, path)
                -> builds _item_ranges_by_file: {s3_uri: [ItemRange, ...]}
            -> create_stream(path, "rb") -> __call__(client, meta)
                -> .metadata files: falls back to a sequential() reader (no ranges available)
                -> .distcp files: DcpPlannedS3Reader using the ranges for that file, or a
                   range_based() reader if no ranges are known for it (should not normally happen)
    """

    def __init__(self, max_gap_size: Union[int, float] = DEFAULT_MAX_GAP_SIZE,
                 max_buffers: Optional[int] = None, prefetch_depth: int = 1) -> None:
        self._item_ranges_by_file: Dict[str, List[ItemRange]] = {}
        self._max_gap_size = max_gap_size
        self._max_buffers = max_buffers
        self._prefetch_depth = prefetch_depth

    def set_item_ranges_by_file(
            self, plan_items: "List[ReadItem]",
            storage_data: "Dict[MetadataIndex, _StorageInfo]",
            base_path: Union[str, os.PathLike]) -> None:
        """Extract and store item ranges per file from a DCP load plan. Called by
        S3StorageReader.prepare_local_plan() to inject range metadata ahead of read_data()."""
        self._item_ranges_by_file = {}
        for read_item in plan_items:
            item_md = storage_data[read_item.storage_index]
            s3_uri = os.path.join(base_path, item_md.relative_path)
            self._item_ranges_by_file.setdefault(s3_uri, []).append(
                ItemRange(item_md.offset, item_md.offset + item_md.length))

    def __call__(
            self, client: S3RdmaClient, meta: S3BucketKey, *,
            read_range: Optional[Tuple[int, int]] = None) -> S3Reader:
        if meta.key.endswith(".metadata"):
            return S3ReaderConstructor.sequential()(client, meta, read_range=read_range)

        s3_uri = f"s3://{meta.bucket}/{meta.key}"
        item_ranges = self._item_ranges_by_file.get(s3_uri)
        if not item_ranges or all(r.start == r.end for r in item_ranges):
            # No ranges known for this file, or every recorded range is zero-length (e.g. an
            # empty tensor) - DcpPlannedS3Reader's coalescer requires at least one non-empty
            # range, so fall back to a normal reader rather than failing the load.
            return S3ReaderConstructor.range_based()(client, meta, read_range=read_range)

        return DcpPlannedS3Reader(
            client, meta, item_ranges, read_range=read_range, max_gap_size=self._max_gap_size,
            max_buffers=self._max_buffers, prefetch_depth=self._prefetch_depth)
