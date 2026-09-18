#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import uuid

import lightning
import torch

from s3cutorchconnector.s3client import S3RdmaClient
from s3cutorchconnector.lightning import S3LightningCheckpoint


# Live-server round trip: save_checkpoint/load_checkpoint work normally with no weights_only
# override, exercising the actual S3 write/read path (the weights_only tests below mock
# torch.load, so this is the only test that touches real data).
def test_lightning_checkpoint_save_and_load_round_trip():
    client = S3RdmaClient()
    bucket_name = f"test-lightning-ckpt-{uuid.uuid4()}"
    key = "epoch0.ckpt"
    s3_uri = f"s3://{bucket_name}/{key}"
    client.s3_client.create_bucket(Bucket=bucket_name)
    checkpoint = S3LightningCheckpoint()

    try:
        state = {"epoch": 3, "state_dict": {"weight": torch.arange(10, dtype=torch.float32)}}
        checkpoint.save_checkpoint(state, s3_uri)

        loaded = checkpoint.load_checkpoint(s3_uri)
        assert loaded["epoch"] == state["epoch"]
        assert torch.equal(loaded["state_dict"]["weight"], state["state_dict"]["weight"])
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key=key)
        client.s3_client.delete_bucket(Bucket=bucket_name)


# The next three tests mock torch.load to capture the weights_only value it was actually called
# with, rather than relying on load behavior differences - they never touch S3 (an S3Reader is
# constructed but torch.load, the only thing that would read from it, is replaced).
def _load_checkpoint_capturing_weights_only(monkeypatch, lightning_version, **load_kwargs):
    monkeypatch.setattr(lightning, "__version__", lightning_version)
    captured = {}

    def fake_torch_load(*_args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    checkpoint = S3LightningCheckpoint()
    checkpoint.load_checkpoint("s3://some-bucket/some-key", **load_kwargs)
    return captured["weights_only"]


def test_load_checkpoint_defaults_weights_only_false_below_lightning_2_6(monkeypatch):
    assert _load_checkpoint_capturing_weights_only(monkeypatch, "2.5.0") is False


def test_load_checkpoint_leaves_weights_only_none_from_lightning_2_6(monkeypatch):
    assert _load_checkpoint_capturing_weights_only(monkeypatch, "2.6.0") is None


def test_load_checkpoint_respects_explicit_weights_only_regardless_of_lightning_version(monkeypatch):
    assert _load_checkpoint_capturing_weights_only(monkeypatch, "2.5.0", weights_only=True) is True
    assert _load_checkpoint_capturing_weights_only(monkeypatch, "2.6.0", weights_only=False) is False
