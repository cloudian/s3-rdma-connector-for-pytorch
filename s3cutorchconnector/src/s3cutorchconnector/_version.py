#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import importlib.metadata

# __package__ is 's3cutorchconnector'; read the version from the installed distribution's
# metadata so it always matches pyproject.toml's [project] version with no duplication.
__version__ = importlib.metadata.version(__package__)
