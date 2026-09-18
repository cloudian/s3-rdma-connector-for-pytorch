#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

# Original source code:
#   https://github.com/awslabs/s3-connector-for-pytorch/blob/main/s3torchconnector/src/s3torchconnector/dcp/s3_prefix_strategy.py
# Pure filename metadata generation, no RDMA-specific behavior - ported as-is.

# pylint: disable=too-few-public-methods
# Each strategy class here intentionally exposes one method (generate_prefix); that's the point
# of the strategy pattern, not a design smell.

from abc import ABC, abstractmethod
from typing import List, Optional

import torch.distributed as dist


class S3PrefixStrategyBase(ABC):
    """Base class for S3 prefix generation strategies, used by S3StorageWriter to spread
    checkpoint shard keys across S3 key-prefixes and avoid single-prefix request throttling."""

    # Delegates to generate_prefix so a strategy instance can be called directly.
    def __call__(self, rank: int) -> str:
        return self.generate_prefix(rank)

    # Generate the storage prefix for the given rank.
    @abstractmethod
    def generate_prefix(self, rank: int) -> str:
        ...


class DefaultPrefixStrategy(S3PrefixStrategyBase):
    """Default strategy: same rank-based naming FileSystemWriter already uses, no extra prefix."""

    # Plain rank-based name, matching FileSystemWriter's own default - no behavior change.
    def generate_prefix(self, rank: int) -> str:
        return f"__{rank}_"


class NumericPrefixStrategy(S3PrefixStrategyBase):
    """Base class for numeric prefix generation strategies. Maps each rank to a distinct
    fixed-length numeric string with its digits reversed, so consecutive ranks land on
    lexicographically distant prefixes - spreading load across S3's internal partitioning."""

    # Validates and stores the numeric-prefix parameters, then precomputes the rank->prefix map.
    def __init__(self, base: int, epoch_num: Optional[int] = None,
                 min_prefix_length: int = 10, prefix_count: Optional[int] = None):
        """
        Args:
            base: The numeric base for the prefix (e.g. 2 for binary, 16 for hex).
            epoch_num: Epoch number for checkpoint ordering. If None, omitted from the prefix.
            min_prefix_length: Minimum length of the generated prefix (zero-padded). Must be positive.
            prefix_count: Number of unique prefixes to generate. Defaults to the distributed
                world size if not given (or 1 if not running distributed).
        """
        # base must be > 1: _calculate_prefix_length's `size *= self.base` never grows otherwise
        # (base 0 or 1), hanging in an infinite loop for any prefix_count > 1 instead of raising.
        if isinstance(base, bool) or not isinstance(base, int) or base < 2:
            raise ValueError(f"Base must be an integer greater than 1, got {base}")
        if min_prefix_length < 1:
            raise ValueError(f"Minimum prefix length must be positive, got {min_prefix_length}")
        # isinstance(x, int) alone would also accept bool (bool is an int subclass in Python).
        if epoch_num is not None and (isinstance(epoch_num, bool) or not isinstance(epoch_num, int)):
            raise ValueError(f"Epoch number must be None or an integer, got {epoch_num}")
        if prefix_count is not None and (
                isinstance(prefix_count, bool) or not isinstance(prefix_count, int) or prefix_count < 1):
            raise ValueError(f"Prefix count must be a positive integer, got {prefix_count}")

        self.base = base
        self.epoch_num = epoch_num
        self.min_prefix_len = min_prefix_length

        self.prefix_count = 1
        if prefix_count is not None:
            self.prefix_count = prefix_count
        elif dist.is_initialized():
            self.prefix_count = dist.get_world_size()

        self.prefix_map = self._generate_prefix_map()

    # Prefix in the form <pattern>/epoch_<num>/__<rank>_, or <pattern>/__<rank>_.
    def generate_prefix(self, rank: int) -> str:
        epoch_suffix = f"epoch_{self.epoch_num}/" if self.epoch_num is not None else ""
        return f"{self.prefix_map[rank % len(self.prefix_map)]}/{epoch_suffix}__{rank}_"

    # Precompute one digit-reversed, zero-padded prefix string per slot in self.prefix_count.
    def _generate_prefix_map(self) -> List[str]:
        minimum_required_length = self._calculate_prefix_length()
        adjusted_prefix_length = max(minimum_required_length, self.min_prefix_len)
        return [self._format_number(i, adjusted_prefix_length)[::-1] for i in range(self.prefix_count)]

    # Minimum digit count needed for self.prefix_count unique values in self.base.
    def _calculate_prefix_length(self) -> int:
        prefix_length = 1
        size = self.base
        while size < self.prefix_count:
            prefix_length += 1
            size *= self.base
        return prefix_length

    # Format a number as a zero-padded string in this strategy's base.
    @abstractmethod
    def _format_number(self, number: int, length: int) -> str:
        ...


class BinaryPrefixStrategy(NumericPrefixStrategy):
    """Binary (base 2) prefix generation strategy, using only 0 and 1."""

    # Fixes the base to 2, forwarding everything else to NumericPrefixStrategy.
    def __init__(self, epoch_num: Optional[int] = None, min_prefix_length: int = 10,
                 prefix_count: Optional[int] = None):
        super().__init__(
            base=2, epoch_num=epoch_num, min_prefix_length=min_prefix_length,
            prefix_count=prefix_count)

    # Zero-padded binary representation of number.
    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}b")


class HexPrefixStrategy(NumericPrefixStrategy):
    """Hexadecimal (base 16) prefix generation strategy."""

    # Fixes the base to 16, forwarding everything else to NumericPrefixStrategy.
    def __init__(self, epoch_num: Optional[int] = None, min_prefix_length: int = 10,
                 prefix_count: Optional[int] = None):
        super().__init__(
            base=16, epoch_num=epoch_num, min_prefix_length=min_prefix_length,
            prefix_count=prefix_count)

    # Zero-padded hexadecimal representation of number.
    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}x")


class RoundRobinPrefixStrategy(S3PrefixStrategyBase):
    """Distributes ranks across a user-provided list of prefixes, round-robin."""

    # Validates the prefix list and stores it (plus the optional epoch number) for later lookup.
    def __init__(self, user_prefixes: List[str], epoch_num: Optional[int] = None):
        if not user_prefixes:
            raise ValueError("user_prefixes must not be empty")
        for prefix in user_prefixes:
            # A leading '/' makes the resulting S3 key absolute, so os.path.join (used to build
            # the final key) silently discards the bucket path instead of prefixing it.
            if not prefix or prefix.startswith("/"):
                raise ValueError(f"user_prefixes entries must be non-empty and not start with '/', got {prefix!r}")
        self.user_prefixes = user_prefixes
        self.epoch_num = epoch_num

    # Prefix in the form <user_prefix>/epoch_<num>/__<rank>_, or <user_prefix>/__<rank>_.
    def generate_prefix(self, rank: int) -> str:
        epoch_suffix = f"epoch_{self.epoch_num}/" if self.epoch_num is not None else ""
        return f"{self.user_prefixes[rank % len(self.user_prefixes)]}/{epoch_suffix}__{rank}_"
