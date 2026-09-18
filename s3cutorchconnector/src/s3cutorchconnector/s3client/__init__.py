import botocore.pycuobjclient

from .s3client import S3RdmaClient
from .s3client_config import S3ClientConfig


RdmaDataPtr = botocore.pycuobjclient.CuDataPtr
RdmaDataBuffer = botocore.pycuobjclient.CuDataBuffer
create_read_buffer = botocore.pycuobjclient.cu_create_read_buffer


__all__ = [
    "S3RdmaClient",
    "S3ClientConfig",
    "RdmaDataPtr",
    "RdmaDataBuffer",
    "create_read_buffer",
]
