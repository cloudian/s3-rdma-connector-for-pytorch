#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from concurrent import futures
from typing import Union

from .s3client import S3RdmaClient
from ._common import parse_s3_uri


class S3Writer(io.BufferedIOBase):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, client: S3RdmaClient, s3_uri: str):
        self.client = client
        self.bucket_name, self.key = parse_s3_uri(s3_uri)
        self.part_size = self.client.s3_client.buffer_pool.buffer_size()
        self._buffer = self.client.s3_client.buffer_pool.get_buffer()
        self._unflushed = 0
        self._pos = 0
        self._multipart = None
        self._parts = []
        self._thread_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(
        self,
        # Ignoring the type for this as Buffer protocol is not supported
        data: Union[bytes, memoryview],  # type: ignore
    ) -> int:
        """Write bytes to BytesIO.

        Args:
            data (bytes | memoryview): bytes to write

        Returns:
            int: Number of bytes written
        """

        off = 0
        data_len = len(data)
        while off < data_len:
            # Start/continue MPU if we've filled the current buffer
            if self._unflushed == self.part_size:
                if self._multipart is None:
                    self._thread_pool = futures.ThreadPoolExecutor(max_workers=self.client.max_concurrent_writes)
                    self._multipart = self.client.s3_client.create_multipart_upload(Bucket=self.bucket_name, Key=self.key)
                    self._parts = []
                self._async_upload_part()
                # Use a fresh buffer for next write
                self._buffer = self.client.s3_client.buffer_pool.get_buffer()
                self._unflushed = 0

            # Copy contents to current buffer
            to_write = min(data_len - off, self.part_size - self._unflushed)
            self._buffer.slice(self._unflushed, to_write).copy_from_buffer(data[off:off + to_write])
            off += to_write
            self._unflushed += to_write

        self._pos += data_len
        return data_len

    def close(self):
        """Close write-stream and write object to S3."""

        # Already closed?
        if self._buffer is None:
            return

        if self._multipart:
            # Upload final part if we have unflushed data
            if self._unflushed != 0:
                self._async_upload_part()

            # Wait for all part uploads to complete
            part_info = {
                            'Parts': list(map(lambda p: p.result(), self._parts)),
                        }
            self.client.s3_client.complete_multipart_upload(Bucket=self.bucket_name, Key=self.key,
                                                            UploadId=self._multipart['UploadId'],
                                                            MultipartUpload=part_info)
            self._thread_pool.shutdown()
        else:
            self.client.s3_client.put_object_direct(self._buffer.slice(0, self._unflushed), Bucket=self.bucket_name, Key=self.key)
            self.client.s3_client.buffer_pool.put_buffer(self._buffer)
        self._buffer = None

    def flush(self):
        """No-op"""

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def _async_upload_part(self):
        buf = self._buffer
        part_len = self._unflushed
        part_number = len(self._parts) + 1
        part = self._thread_pool.submit(lambda: self._upload_part(part_number, buf, part_len))
        self._parts.append(part)

    def _upload_part(self, part_number, buf, part_len):
        part = self.client.s3_client.upload_part_direct(buf.slice(0, part_len), Bucket=self.bucket_name, Key=self.key,
                                                        UploadId=self._multipart['UploadId'], PartNumber=part_number)
        self.client.s3_client.buffer_pool.put_buffer(buf)
        return {
            'PartNumber': part_number,
            'ETag': part['ETag'],
        }
