.PHONY: app build image clean

ifneq ("local", "$(STAGE)")
ifneq (,$(wildcard aws.env))
include aws.env
export
endif
endif

build-app-api:
	@echo "Building the application image..."
	@docker buildx build -t kajima-construction-app-api:latest -f ./docker/app_api/Dockerfile --platform linux/x86_64 --provenance false .

build-batch-processor-api:
	@echo "Building the batch processor image..."
	@docker buildx build -t kajima-construction-batch-processor-api:latest -f ./docker/batch_processor_api/Dockerfile --platform linux/x86_64 --provenance false .

build-all: build-app-api build-batch-processor-api

local-test:
	@echo "Start local development environment..."
	@docker compose -f docker-compose.yml up -d
	@docker compose logs

# Push to AWS
authenticate:
	@echo "Authenticating with AWS ECR..."
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).$(AWS_DOMAIN)

push-app-api: build-app-api
	@echo "Pushing app-api image to AWS ECR"
	@docker tag kajima-construction-app-api:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).$(AWS_DOMAIN)/$(AppApiRepositoryName):latest
	@docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).$(AWS_DOMAIN)/$(AppApiRepositoryName):latest

push-batch-processor-api: build-batch-processor-api
	@echo "Pushing batch-processor-api image to AWS ECR"
	@docker tag kajima-construction-batch-processor-api:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).$(AWS_DOMAIN)/$(BatchProcessorRepositoryName):latest
	@docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).$(AWS_DOMAIN)/$(BatchProcessorRepositoryName):latest

push-all: authenticate push-app-api push-batch-processor-api

# Clean
clean-app:
	@echo "Cleaning up the app api image..."
	@if [ ! -z "$$(docker images -q kajima-construction-app-api:latest 2> /dev/null)" ]; then \
			echo "Removing existing Docker image..."; \
			docker rmi -f kajima-construction-app-api:latest; \
	else \
			echo "No existing image found."; \
	fi

clean-processor:
	@echo "Cleaning up the batch processor api image..."
	@if [ ! -z "$$(docker images -q kajima-construction-batch-processor-api:latest 2> /dev/null)" ]; then \
			echo "Removing existing Docker image..."; \
			docker rmi -f kajima-construction-batch-processor-api:latest; \
	else \
			echo "No existing image found."; \
	fi

clean-all: clean-app clean-processor