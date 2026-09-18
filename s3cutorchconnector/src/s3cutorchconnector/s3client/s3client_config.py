#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class S3ClientConfig:
    """A dataclass exposing configurable parameters for the S3 client.

    Args:
    throughput_target_gbps(float): Throughput target in Gigabits per second (Gbps) that we are trying to reach.
        Currently unused by the rdma s3 client
    part_size(int): Size (bytes) of file parts that will be uploaded/downloaded.
        Note: for saving checkpoints, the inner client will adjust the part size to meet the service limits.
        (max number of parts per upload is 10,000, minimum upload part size is 5 MiB).
        Part size must have values between 5MiB and 5GiB.
        8MiB by default (may change in future).
    force_path_style(bool): forceful path style addressing for S3 client.
    max_attempts(int): number of retry attempts for retryable errors.
    profile(Optional[str]): profile name to use for S3 authentication.
    requester_pays(bool): Not supported by Cloudian HyperStore. Originally intended to enable access to Requester Pays buckets.
    """

    throughput_target_gbps: float = 10.0
    part_size: int = 10 * 1024 * 1024
    unsigned: bool = False
    force_path_style: bool = False
    max_attempts: int = 10
    profile: Optional[str] = None
    requester_pays: bool = False

    # requester_pays is unconditionally unsupported: Cloudian HyperStore does not offer Requester Pays
    # buckets, so any attempt to enable it is rejected outright.
    def __post_init__(self):
        if self.requester_pays:
            raise ValueError(
                "requester_pays=True is not supported by Cloudian HyperStore."
            )
