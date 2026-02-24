import boto3
import src.vars as var

endpoint_url = None
if var.STAGE == "local":
    endpoint_url = "https://localhost.localstack.cloud:4566"
s3_client = boto3.client("s3", endpoint_url=endpoint_url)


def upload_to_s3(
    bucket_name: str,
    key: str,
    data: bytes,
    metadata: dict = None,
    content_type: str = "application/octet-stream",
):
    """Upload data to S3 bucket"""
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata,
        )
        return {"status": "success", "message": f"Uploaded {key} to {bucket_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_from_s3(
    bucket_name: str,
    key: str,
):
    """Delete data from S3 bucket"""
    try:
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=key,
        )
        return {"status": "success", "message": f"Deleted {key} from {bucket_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
