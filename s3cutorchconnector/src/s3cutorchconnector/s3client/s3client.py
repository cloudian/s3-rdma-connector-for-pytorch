from typing import Optional
import os

import botocore.session

from .s3client_config import S3ClientConfig


class S3RdmaClient:
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-instance-attributes
    def __init__(self,
                 region: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 s3client_config: Optional[S3ClientConfig] = None):

        self.region = region
        self.endpoint = endpoint
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.s3client_config = s3client_config or S3ClientConfig()

        config_kwargs = {'retries': {'max_attempts': self.s3client_config.max_attempts}}
        if self.s3client_config.force_path_style:
            config_kwargs['s3'] = {'addressing_style': 'path'}
        config = botocore.client.Config(**config_kwargs)

        session = botocore.session.Session(profile=self.s3client_config.profile)
        self.s3 = session.create_client('s3', region, endpoint_url=endpoint, aws_access_key_id=aws_access_key_id,
                                        aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token,
                                        config=config)
        self.max_concurrent_reads = self.s3.max_concurrent_reads
        self.max_concurrent_writes = int(os.environ.get("S3RDMA_CONCURRENT_WRITES", 4))

    def __deepcopy__(self, _memo):
        return S3RdmaClient(region=self.region, endpoint=self.endpoint,
                            aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key,
                            aws_session_token=self.aws_session_token, s3client_config=self.s3client_config)

    @property
    def s3_client(self):
        return self.s3
