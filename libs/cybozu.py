import os
import requests
from typing import Any
import src.vars as var
from pydantic import BaseModel


class CybozuFile(BaseModel):
    file_key: str
    file_name: str
    file_type: str
    file_content: bytes


def download_image_file_from_kintone(file_key: str) -> bytes | None:
    url = f"https://{var.KINTONE_SUBDOMAIN}.kintone.com/k/v1/file.json"
    headers = {
        "X-Cybozu-API-Token": var.KINTONE_API_TOKEN,
    }
    params = {
        "fileKey": file_key,
    }
    try:
        response = requests.get(url=url, params=params, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        raise e
    return response.content


def get_kintone_record(record_id: str) -> dict[str, Any]:
    url = f"https://{os.environ['KINTONE_SUBDOMAIN']}.kintone.com/k/v1/record.json"
    params = {
        "app": os.environ["KINTONE_APP_ID"],
        "id": record_id,
    }

    # Simplify headers to match your curl command
    headers = {
        "X-Cybozu-API-Token": os.environ["KINTONE_API_TOKEN"],
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        # Print more error details for debugging
        if hasattr(e, "response") and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            print(f"Request headers: {e.request.headers}")
        raise e


def update_kintone_key(record_id: str, update_key: str, update_value: Any):
    url = f"https://{os.environ['KINTONE_SUBDOMAIN']}.kintone.com/k/v1/record.json"
    headers = {
        "X-Cybozu-API-Token": os.environ["KINTONE_API_TOKEN"],
    }
    data = {
        "app": os.environ["KINTONE_APP_ID"],
        "id": record_id,
        "record": {update_key: {"value": update_value}},
    }
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        # Print more error details for debugging
        if hasattr(e, "response") and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            print(f"Request headers: {e.request.headers}")
        raise e
