#!/usr/bin/env python3
'''
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

As a part of enforcing test ownership, we want to maintain a list of existing PyTorch labels
to verify the owners' existence. This script outputs a file containing a list of existing
pytorch/pytorch labels so that the file could be uploaded to S3.

This script assumes the correct env vars are set for AWS permissions.

'''

import boto3  # type: ignore[import]
import json
from typing import List
from gitutils import gh_get_labels


def send_labels_to_S3(labels: List[str]) -> None:
    labels_file_name = "pytorch_labels.json"
    obj = boto3.resource('s3').Object('ossci-metrics', labels_file_name)
    obj.put(Body=json.dumps(labels).encode())


def main() -> None:
    send_labels_to_S3(gh_get_labels("pytorch", "pytorch"))


if __name__ == '__main__':
    main()
