#!/bin/bash
export AWS_DEFAULT_REGION=ap-northeast-1

AWS_REGION=ap-northeast-1
AWS_ACCOUNT_ID=000000000000
AWS_DOMAIN=localhost.localstack.cloud:4566
AppApiRepositoryName=localstack-kajima-app-api
BatchProcessorRepositoryName=localstack-kajima-batch-processor-api

export AWS_REGION
export AWS_ACCOUNT_ID
export AWS_DOMAIN
export AppApiRepositoryName
export BatchProcessorRepositoryName

PROJECT_NAME=kajima
ENVIRONMENT=dev

echo "Creating s3 bucket..."
awslocal s3 mb "s3://localstack-${PROJECT_NAME}-uploaded-image"

echo "Creating sqs queue..."
awslocal sqs create-queue --queue-name "localstack-${PROJECT_NAME}-image-queue"
SQS_QUEUE_URL=$(awslocal sqs get-queue-url --queue-name "localstack-${PROJECT_NAME}-image-queue" | jq -r '.QueueUrl')
export SQS_QUEUE_URL

echo "Creating ECR repository..."
awslocal ecr create-repository \
      --repository-name "${AppApiRepositoryName}" \
      --image-scanning-configuration scanOnPush=true
awslocal ecr create-repository \
      --repository-name "${BatchProcessorRepositoryName}" \
      --image-scanning-configuration scanOnPush=true

echo "Building docker image..."
make build-all
make push-all

echo "Creating lambda function..."
APP_API_IMAGE_URI="$(awslocal ecr describe-repositories --repository-names ${AppApiRepositoryName} | jq -r '.repositories[0].repositoryUri'):latest"
BATCH_PROCESSOR_API_IMAGE_URI="$(awslocal ecr describe-repositories --repository-names ${BatchProcessorRepositoryName} | jq -r '.repositories[0].repositoryUri'):latest"

# Read environment variables from .env file
ENV_VARS=""
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  # Convert .env file to comma-separated KEY=VALUE format for Lambda
  ENV_VARS=$(grep -v '^#' .env | grep '=' | sed 's/\r$//' | awk -F= '{print $1"="$2}' | paste -sd "," -)
  # Add S3 and SQS variables
  ENV_VARS="${ENV_VARS},S3_BUCKET_NAME=localstack-${PROJECT_NAME}-uploaded-image,SQS_QUEUE_URL=${SQS_QUEUE_URL},STAGE=local"
else
  echo "No .env file found, using default environment variables"
  ENV_VARS="S3_BUCKET_NAME=localstack-${PROJECT_NAME}-uploaded-image,SQS_QUEUE_URL=${SQS_QUEUE_URL},STAGE=local"
fi

awslocal lambda create-function \
      --function-name "localstack-${PROJECT_NAME}-app-api" \
      --runtime python3.12 \
      --timeout 30 \
      --package-type Image \
      --code ImageUri=$APP_API_IMAGE_URI \
      --role arn:aws:iam::000000000000:role/lambda-role \
      --environment "Variables={$ENV_VARS}"


awslocal lambda create-function \
      --function-name "localstack-${PROJECT_NAME}-batch-processor-api" \
      --runtime python3.12 \
      --timeout 900 \
      --package-type Image \
      --code ImageUri=$BATCH_PROCESSOR_API_IMAGE_URI \
      --role arn:aws:iam::000000000000:role/lambda-role \
      --environment "Variables={$ENV_VARS,LAMBDA_RUNTIME_ENVIRONMENT_TIMEOUT=100}" \
      --memory-size 1024

echo "Creating event source mapping..."

SQS_QUEUE_ARN=$(awslocal sqs get-queue-attributes --queue-url $SQS_QUEUE_URL --attribute-names QueueArn | jq -r '.Attributes.QueueArn')
PROCESSOR_LAMBDA_NAME="localstack-${PROJECT_NAME}-batch-processor-api"
SQS_BATCH_SIZE=4
awslocal lambda create-event-source-mapping \
  --event-source-arn $SQS_QUEUE_ARN \
  --function-name $PROCESSOR_LAMBDA_NAME \
  --batch-size $SQS_BATCH_SIZE \
  --maximum-batching-window-in-seconds 30 \
  --function-response-types ReportBatchItemFailures


echo "Finish"