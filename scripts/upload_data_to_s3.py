"""
Upload POC Dataset to S3

This script uploads the local dataset to S3 so SageMaker can use it.

Usage:
    # First create the local dataset
    python scripts/prepare_minimal_poc.py --output ./data --num-images 100
    
    # Then upload to S3
    python scripts/upload_data_to_s3.py --local-dir ./data --bucket tdsp-data-products-dev --prefix tdspds/rust-detection/data
"""

import argparse
import os

import boto3
from botocore.exceptions import ClientError


def upload_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str):
    """
    Upload a local directory to S3.
    
    Args:
        local_dir: Local directory path
        bucket: S3 bucket name
        s3_prefix: S3 prefix (folder path)
    """
    s3_client = boto3.client("s3")
    
    uploaded_files = 0
    
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            
            # Calculate S3 key
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            try:
                s3_client.upload_file(local_path, bucket, s3_key)
                uploaded_files += 1
                
                if uploaded_files % 10 == 0:
                    print(f"Uploaded {uploaded_files} files...")
                    
            except ClientError as e:
                print(f"Error uploading {local_path}: {e}")
    
    return uploaded_files


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to S3")
    parser.add_argument("--local-dir", default="./data", help="Local data directory")
    parser.add_argument("--bucket", default="tdsp-data-products-dev", help="S3 bucket")
    parser.add_argument("--prefix", default="tdspds/rust-detection/data", help="S3 prefix")
    
    args = parser.parse_args()
    
    print(f"Uploading {args.local_dir} to s3://{args.bucket}/{args.prefix}/")
    print()
    
    # Upload train data
    train_dir = os.path.join(args.local_dir, "train")
    if os.path.exists(train_dir):
        print("Uploading training data...")
        count = upload_directory_to_s3(train_dir, args.bucket, f"{args.prefix}/train")
        print(f"  Uploaded {count} training files")
    
    # Upload test data
    test_dir = os.path.join(args.local_dir, "test")
    if os.path.exists(test_dir):
        print("Uploading test data...")
        count = upload_directory_to_s3(test_dir, args.bucket, f"{args.prefix}/test")
        print(f"  Uploaded {count} test files")
    
    # Upload metadata
    metadata_path = os.path.join(args.local_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        print("Uploading metadata...")
        s3_client = boto3.client("s3")
        s3_client.upload_file(metadata_path, args.bucket, f"{args.prefix}/metadata/metadata.csv")
        print("  Uploaded metadata.csv")
    
    print()
    print("=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print()
    print("S3 Structure:")
    print(f"  s3://{args.bucket}/{args.prefix}/")
    print(f"  ├── train/")
    print(f"  │   ├── no_rust/*.jpg")
    print(f"  │   └── rust/*.jpg")
    print(f"  ├── test/")
    print(f"  │   ├── no_rust/*.jpg")
    print(f"  │   └── rust/*.jpg")
    print(f"  └── metadata/")
    print(f"      └── metadata.csv")
    print()
    print("Update your pipelines-config.yml with these S3 URIs:")
    print()
    print("InputDataConfig:")
    print(f'  - ChannelName: "train"')
    print(f'    S3Uri: "s3://{args.bucket}/{args.prefix}/train"')
    print(f'  - ChannelName: "test"')
    print(f'    S3Uri: "s3://{args.bucket}/{args.prefix}/test"')
    print(f'  - ChannelName: "metadata"')
    print(f'    S3Uri: "s3://{args.bucket}/{args.prefix}/metadata"')


if __name__ == "__main__":
    main()
