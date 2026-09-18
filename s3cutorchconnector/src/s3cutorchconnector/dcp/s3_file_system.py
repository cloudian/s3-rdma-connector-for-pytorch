#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

# Original source code:
#   https://github.com/awslabs/s3-connector-for-pytorch/blob/main/s3torchconnector/src/s3torchconnector/dcp/s3_file_system.py
# Adapted to use s3 rdma client

import dataclasses
import io
import logging
import os
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Union, Optional

from botocore.exceptions import ClientError

from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    wait_random_exponential,
)
from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
    FileSystemBase,
)
from torch.distributed.checkpoint.planner import LoadPlan, SavePlan

from ..s3client import S3RdmaClient
from ..s3writer import S3Writer
from .._common import parse_s3_uri
from .._types import S3BucketKey
from ..reader_constructor import S3ReaderConstructor, S3ReaderConstructorProtocol, DcpOptimizedConstructor
from .s3_prefix_strategy import S3PrefixStrategyBase, DefaultPrefixStrategy

logger = logging.getLogger(__name__)


class S3FileSystem(FileSystemBase):
    def __init__(self, region: Optional[str] = None, s3_client: Optional[S3RdmaClient] = None,
                 reader_constructor: Optional[S3ReaderConstructorProtocol] = None) -> None:
        self._path: Union[str, os.PathLike] = ""
        self._client = (
            s3_client
            if s3_client is not None
            else S3RdmaClient(region)
        )
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        """
        Create a stream for reading or writing to S3.

        Args:
            path (Union[str, os.PathLike]): The S3 path to read or write.
            mode (str): The mode for the stream. Supports 'rb' for read mode and 'wb' for write mode.

        Yields:
            io.BufferedIOBase: A stream for reading or writing to S3.

        Raises:
            ValueError: If the mode is not 'rb' or 'wb'.
        """
        path_str = _path_or_str_to_str(path)

        # The original AWS code uses custom reader and writers.
        # We'll reuse the readers and writers from the dataset iterator and checkpoiting code
        if mode == "wb":  # write mode
            logger.debug("create_stream writable for %s", path_str)
            writer = S3Writer(self._client, path_str)
            yield writer
            writer.close()
        elif mode == "rb":  # read mode
            logger.debug("create_stream readable for %s", path_str)
            bucket, key = parse_s3_uri(path_str)
            meta = S3BucketKey(bucket, key)
            reader = self._reader_constructor(self._client, meta)
            yield reader
            reader.close()
        else:
            raise ValueError(
                f"Invalid {mode=} mode argument: create_stream only supports rb (read mode) & wb (write mode)"
            )

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> str:
        """
        Concatenate a suffix to the given path.

        Args:
            path (Union[str, os.PathLike]): The base path.
            suffix (str): The suffix to concatenate.

        Returns:
            str: The concatenated path.
        """
        logger.debug("concat paths %s and %s", path, suffix)
        path_str = os.fspath(path)
        result = os.path.join(path_str, suffix)
        return result

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """
        Initialize the path for the filesystem.

        Args:
            path (Union[str, os.PathLike]): The path to initialize.

        Returns:
            Union[str, os.PathLike]: The initialized path.
        """
        logger.debug("init_path for %s", path)
        self._path = path
        return self._path

    # pylint: disable=arguments-renamed
    def rename(
        self, old_path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """Rename an object in S3.

        This is emulated by copying it to a new path and deleting the old path. The deletion part is retried (see also
        :func:`S3FileSystem._delete_with_retry`).

        Args:
            old_path (Union[str, os.PathLike]): The current path of the object.
            new_path (Union[str, os.PathLike]): The new path for the object.

        Raises:
            ValueError: If the old and new paths point to different buckets.
            S3Exception: If there is an error with the S3 client.
        """
        logger.debug("rename %s to %s", old_path, new_path)

        old_path_str = _path_or_str_to_str(old_path)
        new_path_str = _path_or_str_to_str(new_path)

        old_bucket, old_key = parse_s3_uri(old_path_str)
        escaped_old_key = self._escape_path(old_key)
        logger.debug("rename: escaped version of the source key: %s", escaped_old_key)
        new_bucket, new_key = parse_s3_uri(new_path_str)

        if old_bucket != new_bucket:
            raise ValueError(
                "Source and destination buckets cannot be different (rename does not support cross-buckets operations)"
            )

        self._client.s3.copy_object(
            CopySource={'Bucket': old_bucket, 'Key': escaped_old_key},
            Bucket=new_bucket,
            Key=new_key,
        )
        logger.debug("rename: copied %s to %s successfully", old_path_str, new_path_str)
        self._delete_with_retry(old_bucket, old_key)
        logger.debug("rename: s3://%s/%s successfully", old_bucket, old_key)

    def mkdir(self, _path: Union[str, os.PathLike]) -> None:
        """No-op method for creating directories in S3 (not needed)."""

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        logger.debug("exists %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise
            return False
        return True

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        logger.debug("remove %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.s3.delete_object(Bucket=bucket, Key=key)
        except ClientError:
            logger.exception("Failed to remove object from S3")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        logger.debug("validate_checkpoint_id for %s", checkpoint_id)

        if isinstance(checkpoint_id, Path):
            return True

        try:
            parse_s3_uri(_path_or_str_to_str(checkpoint_id))
        except ValueError:
            return False
        return True

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.ERROR),
        reraise=True,
    )
    def _delete_with_retry(self, bucket_name: str, old_key: str):
        """Wrapper around :func:`S3Client.delete_object` to retry the deletion.

        Will retry a maximum of 3 times, only for `S3Exception`s, and wait between retries. It will reraise the caught
        exception too, and logs retries and final error, if any."""
        self._client.s3.delete_object(Bucket=bucket_name, Key=old_key)

    @staticmethod
    def _escape_path(string):
        """URL-encodes path segments while preserving '/' separators using urllib.parse.quote().

        Args:
            string (str): URL path string to escape

        Returns:
            str: Path string with each segment percent-encoded, separators preserved
        """
        if not string:
            return string
        parts = []
        for part in string.split("/"):
            parts.append(urllib.parse.quote(part, safe=""))
        return "/".join(parts)


@dataclass
class StorageMetadata:
    """Per-rank S3 key prefix, consumed by FileSystemWriter.write_data() via storage_data.prefix."""
    prefix: str


# pylint: disable=too-many-ancestors,too-few-public-methods
class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        path: str,
        prefix_strategy: Optional[S3PrefixStrategyBase] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            prefix_strategy (Optional[S3PrefixStrategyBase]): Strategy for generating a per-rank
                S3 key prefix, to spread checkpoint shard keys across S3 partitions and avoid
                single-prefix request throttling at scale. Defaults to DefaultPrefixStrategy,
                which uses the same rank-based naming FileSystemWriter already uses (no change
                in behavior).
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=path,
            sync_files=False,
            **kwargs,
        )
        self.fs = S3FileSystem(region)  # type: ignore
        self.path = self.fs.init_path(path)
        self.prefix_strategy = prefix_strategy or DefaultPrefixStrategy()

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """ Attach a per-rank S3 key prefix (from prefix_strategy) to each save plan.

        Only fills in storage_data when the base class's prepare_local_plan hasn't already set
        it - mirroring FileSystemWriter's own prepare_global_plan exactly. With
        use_collectives=False, prepare_local_plan already stamps each rank's plan with its real
        rank before prepare_global_plan ever runs, and (per-rank) plans here is a single-element
        list - so unconditionally reassigning storage_data from enumerate(plans) would apply
        prefix_strategy(0) to every rank, colliding all ranks onto the same shard names. """
        return [
            plan if plan.storage_data is not None
            else dataclasses.replace(plan, storage_data=StorageMetadata(self.prefix_strategy(idx)))
            for idx, plan in enumerate(plans)
        ]

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


# pylint: disable=too-many-ancestors,too-few-public-methods
class S3StorageReader(FileSystemReader):
    def __init__(self, region: str, path: Union[str, os.PathLike],
                 reader_constructor: Optional[S3ReaderConstructorProtocol] = None) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
            reader_constructor (Optional[S3ReaderConstructorProtocol]): Reader constructor created
                using S3ReaderConstructor. Defaults to S3ReaderConstructor.range_based(), so that
                only the byte ranges a rank actually reads are fetched from S3, rather than the
                whole checkpoint shard file.
        """
        super().__init__(path)
        self._reader_constructor = reader_constructor or S3ReaderConstructor.range_based()
        self.fs = S3FileSystem(region, reader_constructor=self._reader_constructor)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Sorts load items by storage offset, so that reads within each per-file stream proceed
        sequentially - required for our LRU-paged reader to avoid re-fetching evicted chunks.
        Also injects range metadata into a dcp_optimized() reader_constructor, if in use."""
        plan.items.sort(key=lambda item: self.storage_data[item.storage_index].offset)
        if isinstance(self._reader_constructor, DcpOptimizedConstructor):
            self._reader_constructor.set_item_ranges_by_file(plan.items, self.storage_data, self.path)
        return plan


def _path_or_str_to_str(path: Union[str, os.PathLike]) -> str:
    return path if isinstance(path, str) else str(path)
