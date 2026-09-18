#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import s3cutorchconnector


def test_version_is_exported_and_nonempty():
    assert isinstance(s3cutorchconnector.__version__, str)
    assert s3cutorchconnector.__version__
    assert "__version__" in s3cutorchconnector.__all__


def test_s3client_config_is_exported_from_package_root():
    assert s3cutorchconnector.S3ClientConfig is s3cutorchconnector.s3client.S3ClientConfig
    assert "S3ClientConfig" in s3cutorchconnector.__all__
