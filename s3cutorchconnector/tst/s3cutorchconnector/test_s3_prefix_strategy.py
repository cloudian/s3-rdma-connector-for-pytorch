#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest

from s3cutorchconnector.dcp import (
    DefaultPrefixStrategy,
    NumericPrefixStrategy,
    BinaryPrefixStrategy,
    HexPrefixStrategy,
    RoundRobinPrefixStrategy,
)


# pylint: disable-next=too-few-public-methods
class _DecimalPrefixStrategy(NumericPrefixStrategy):
    """Minimal concrete NumericPrefixStrategy subclass, for exercising base validation directly -
    BinaryPrefixStrategy/HexPrefixStrategy always pass a fixed, valid base."""

    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}d")


def test_default_prefix_strategy():
    strategy = DefaultPrefixStrategy()
    assert strategy(0) == "__0_"
    assert strategy(5) == "__5_"


def test_binary_prefix_strategy_reverses_digits_per_rank():
    strategy = BinaryPrefixStrategy(prefix_count=4, min_prefix_length=2)
    assert strategy(0) == "00/__0_"
    assert strategy(1) == "10/__1_"
    assert strategy(2) == "01/__2_"
    assert strategy(3) == "11/__3_"


def test_binary_prefix_strategy_wraps_ranks_beyond_prefix_count():
    # The leading numeric prefix segment wraps around prefix_count, but the trailing __{rank}_
    # always reflects the real rank, so full prefixes for rank r and r+prefix_count differ only
    # in that suffix.
    strategy = BinaryPrefixStrategy(prefix_count=2, min_prefix_length=1)
    assert strategy(0) == "0/__0_"
    assert strategy(2) == "0/__2_"
    assert strategy(1) == "1/__1_"
    assert strategy(3) == "1/__3_"


def test_hex_prefix_strategy():
    strategy = HexPrefixStrategy(prefix_count=16, min_prefix_length=1)
    assert strategy(10) == "a/__10_"


def test_numeric_prefix_strategy_includes_epoch():
    strategy = BinaryPrefixStrategy(epoch_num=5, prefix_count=2, min_prefix_length=1)
    assert strategy(0) == "0/epoch_5/__0_"
    assert strategy(1) == "1/epoch_5/__1_"


def test_numeric_prefix_strategy_pads_to_min_length():
    strategy = BinaryPrefixStrategy(prefix_count=2, min_prefix_length=5)
    assert strategy(0) == "00000/__0_"
    assert strategy(1) == "10000/__1_"


@pytest.mark.parametrize("kwargs", [
    {"min_prefix_length": 0},
    {"epoch_num": "not-an-int"},
    {"prefix_count": 0},
    {"prefix_count": "not-an-int"},
    {"epoch_num": True},  # bool is an int subclass in Python; must still be rejected
    {"prefix_count": True},
])
def test_numeric_prefix_strategy_rejects_invalid_args(kwargs):
    with pytest.raises(ValueError):
        BinaryPrefixStrategy(**kwargs)


def test_round_robin_prefix_strategy():
    strategy = RoundRobinPrefixStrategy(["a", "b", "c"])
    assert strategy(0) == "a/__0_"
    assert strategy(1) == "b/__1_"
    assert strategy(3) == "a/__3_"  # wraps around


def test_round_robin_prefix_strategy_with_epoch():
    strategy = RoundRobinPrefixStrategy(["a", "b"], epoch_num=2)
    assert strategy(0) == "a/epoch_2/__0_"


def test_round_robin_prefix_strategy_rejects_empty_list():
    with pytest.raises(ValueError):
        RoundRobinPrefixStrategy([])


@pytest.mark.parametrize("user_prefixes", [["/absolute"], ["ok", ""]])
def test_round_robin_prefix_strategy_rejects_invalid_prefixes(user_prefixes):
    # A leading '/' (or an empty entry) would make the resulting S3 key absolute, which
    # os.path.join silently turns into dropping the bucket path instead of prefixing it.
    with pytest.raises(ValueError):
        RoundRobinPrefixStrategy(user_prefixes)


@pytest.mark.parametrize("base", [0, 1, -2, True, 2.5, "2"])
def test_numeric_prefix_strategy_rejects_invalid_base(base):
    # base <= 1 would otherwise hang _calculate_prefix_length in an infinite loop (size *= base
    # never grows) for any prefix_count > 1, instead of raising here at construction time.
    with pytest.raises(ValueError):
        _DecimalPrefixStrategy(base=base, prefix_count=1, min_prefix_length=1)
