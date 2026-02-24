from PIL import Image
import torch
import src.vars as var
from src.config import Config
from src.model import get_model
from src.utils import get_current_datetime
from aws_lambda_powertools.logging import Logger
from src.schema import ImageRecord
from concurrent.futures import ThreadPoolExecutor
from libs.cybozu import download_image_file_from_kintone, update_kintone_key
from libs.aws_s3 import upload_to_s3
import numpy as np
from io import BytesIO
import os
import time

logger = Logger(service="kajima-processor-api")


def get_model_and_transform():
    logger.info("Loading model and image transforms for batch processing")
    start_time = time.time()

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"

    config_path = f"{os.environ['LAMBDA_TASK_ROOT']}/{var.MODEL_CONFIG_PATH}"
    logger.info(f"Using device: {device}. Config path: {config_path}")
    model_config = Config.from_json(config_path)
    model, image_transforms = get_model(model_config, device=device, test_mode=True)

    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

    return model, image_transforms, device


model, image_transforms, device = get_model_and_transform()


def download_image(idx, file_key: str):
    try:
        file_content = download_image_file_from_kintone(file_key)
        return idx, file_content
    except Exception as e:
        logger.error(f"Error downloading image {file_key}: {str(e)}")
        return idx, None


def download_images_parallel(keys: list[str]):
    """Download multiple images in parallel"""
    results = [None] * len(keys)

    with ThreadPoolExecutor(max_workers=min(2, len(keys))) as executor:
        futures = [
            executor.submit(download_image, idx, key) for idx, key in enumerate(keys)
        ]
        for future in futures:
            if future.exception() is None:
                idx, data = future.result()
                if data is not None:
                    results[idx] = data

    return results


def batch_record_handler(records: list[ImageRecord], record_id: str):
    logger.info(f"Downloading {len(records)} images from Cybozu")
    image_datas = download_images_parallel([record.image_id for record in records])
    if not all(image_datas):
        logger.info(
            f"Failed to download {len(records) - sum(1 for data in image_datas if data)} images"
        )
        logger.info(
            f"Failed at image file keys: {[records[idx].image_id for idx, data in enumerate(image_datas) if data is None]}"
        )
        return {
            "statusCode": 500,
            "content": "Failed to download images",
        }
    try:
        all_predictions = []
        with torch.no_grad():
            BATCH_SIZE = 4
            logger.info(f"Running inference on batch of {BATCH_SIZE} images")
            for i in range(0, len(image_datas), BATCH_SIZE):
                image_batchs = image_datas[i : i + BATCH_SIZE]
                input_tensors = []
                for j, image_data in enumerate(image_batchs):
                    try:
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        image = np.array(image)
                        image = image_transforms(image)
                        input_tensors.append(image)
                    except Exception as ex:
                        logger.error(
                            f"Fail to process image {records[i * BATCH_SIZE + j].image_id}: {str(ex)}"
                        )

                inputs = torch.stack(input_tensors).to(device)
                outputs = model(inputs)
                for _, output in enumerate(outputs):
                    label = torch.argmax(output).item()
                    confidence = torch.softmax(output, dim=0)[label].item()
                    all_predictions.append(
                        {
                            "label": str(label + 1),
                            "confidence": confidence,
                        }
                    )

        update_kintone_key(
            record_id=record_id,
            update_key=var.KINTONE_UPDATE_FIELD_CODE,
            update_value=",".join([f"7-{prediction['label']}" for prediction in all_predictions]),
        )
        # Store result
        processed_idx = []
        for i in range(len(all_predictions)):
            record = records[i]
            prediction = all_predictions[i]
            img_data = image_datas[i]
            try:
                # Store prediction in output S3 bucket
                output_key = f"{record.image_id}_{record.user_id}_{record.record_id}_{get_current_datetime()}_7-{prediction['label']}"
                logger.info(
                    f"Storing processed image to S3: {var.S3_BUCKET_NAME}/{output_key}"
                )
                response = upload_to_s3(
                    bucket_name=var.S3_BUCKET_NAME,
                    key=output_key,
                    data=img_data,
                    content_type=record.content_type,
                    metadata={
                        "prediction": prediction["label"],
                        "confidence": str(prediction["confidence"]),
                        "user_id": record.user_id,
                        "record_id": record.record_id,
                    },
                )
                if response["status"] != "success":
                    raise RuntimeError(response["message"])

                logger.info(
                    f"Successfully processed image {output_key} with prediction {prediction['label']}"
                )
                processed_idx.append(i)
            except Exception as ex:
                logger.error(
                    f"Error storing output of {record.s3_bucket_key}: {str(ex)}"
                )
        logger.info(f"Processed {len(processed_idx)} images successfully")
    except Exception as ex:
        logger.error(f"Error during classification {ex}")

    return {
        "statusCode": 200,
    }


def lambda_handler(event, context):
    logger.info(f"Lambda receive events: {event}")
    body = event.get("body")
    logger.info(f"Received event: {body}")

    data = body["data"]
    user_id = body["user_id"]
    record_id = body["record_id"]

    field_info = data["record"][var.KINTONE_DATA_FIELD_CODE]
    image_records = []

    if field_info["type"] != "FILE":
        logger.error("Field type is not FILE")
        return {
            "statusCode": 400,
            "body": "Field type is not FILE",
        }
    else:
        for file_info in field_info["value"]:
            if file_info["contentType"] in ["image/jpeg", "image/png"]:
                image_records.append(
                    ImageRecord(
                        image_id=file_info["fileKey"],
                        user_id=user_id,
                        record_id=record_id,
                        content_type=file_info["contentType"],
                    )
                )

    logger.info(f"Process {len(image_records)} image files")

    if not image_records:
        logger.error("No valid image files found")
        return {
            "statusCode": 400,
            "body": "No valid image files found",
        }
    return batch_record_handler(image_records, record_id)
