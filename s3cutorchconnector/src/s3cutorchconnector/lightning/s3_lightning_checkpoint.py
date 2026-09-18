#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Optional, Dict, Any

from packaging import version

import lightning
import torch

from lightning.pytorch.plugins.io import CheckpointIO


from .._common import parse_s3_uri
from ..s3client import S3ClientConfig, S3RdmaClient
from ..s3reader import S3Reader
from ..s3writer import S3Writer
from .._types import S3BucketKey


class S3LightningCheckpoint(CheckpointIO):
    """A checkpoint manager for S3 using the :class:`CheckpointIO` interface."""

    # pylint: disable=too-many-function-args
    def __init__(
            self,
            region: Optional[str] = None,
            s3client_config: Optional[S3ClientConfig] = None,
            endpoint: Optional[str] = None):
        self._client = S3RdmaClient(region, endpoint, s3client_config=s3client_config)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
        storage_options: Optional[Any] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and upload to S3.

        Args:
            checkpoint (Dict[str, Any]): Containing model and trainer state
            path (str): Write-target S3 uri
            storage_options: Optional parameters when saving the model/training states.
        """
        self._validate_path(path)
        with S3Writer(self._client, path) as s3objwriter:
            torch.save(checkpoint, s3objwriter)

    def load_checkpoint(
        self,
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
        map_location: Optional[Any] = None,
        weights_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint from an S3 location when resuming or loading ckpt for test/validate/predict stages.

        Args:
            path (str): S3 uri to checkpoint
            map_location: A function, :class:`torch.device`, string or a dict specifying how to remap storage locations.
            weights_only: If True, only loads tensors and primitive types (safer). If False, allows loading
                arbitrary Python objects (less secure). If None, uses PyTorch Lightning's default behavior.
                See https://docs.pytorch.org/docs/main/notes/serialization.html for details.

        Returns:
            Dict[str, Any]: The loaded checkpoint
        """
        self._validate_path(path)
        bucket, key = parse_s3_uri(path)
        bucket_key = S3BucketKey(bucket, key)
        s3objreader = S3Reader(self._client, bucket_key)

        # Maintain backward compatibility: default to False for Lightning <2.6, None for >=2.6.
        # - Lightning >=2.6 lets PyTorch decide the default; weights_only can be set through
        #   Trainer.{fit,validate,test,predict} instead.
        # - Lightning <2.6 always defaulted to weights_only=False:
        #   https://github.com/Lightning-AI/pytorch-lightning/blob/release/2.5.x/src/lightning/fabric/utilities/cloud_io.py#L37
        if weights_only is None:
            if version.parse(lightning.__version__) < version.parse("2.6.0"):
                weights_only = False

        # torch.load() before 2.4 typed weights_only as a plain bool (not Optional), but None
        # acts as False in its internal `if weights_only:` checks, so passing it through is safe
        # for backward compatibility even on older torch.
        return torch.load(s3objreader, map_location, weights_only=weights_only)  # type: ignore

    def remove_checkpoint(
        self,
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
    ) -> None:
        """Remove checkpoint file from the S3 uri.

        Args:
            path (str): S3 uri to checkpoint
        """
        self._validate_path(path)
        bucket, key = parse_s3_uri(path)
        self._client.s3_client.delete_object(Bucket=bucket, Key=key)

    def teardown(self) -> None:
        """No-op"""

    @staticmethod
    def _validate_path(path: str) -> None:
        if not isinstance(path, str):
            raise TypeError(
                f"{type(path).__name__!r} is not a supported type for 'path'. Must be a string formatted as an S3 uri."
            )
