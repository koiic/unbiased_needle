.PHONY: list

AWS_REGION := $(shell eval "aws --profile default configure get region")
IMAGE_TAG := latest ## $(shell eval "git rev-parse --short HEAD")
ECR_NAME := maio-ml_api_repository ## $(shell eval "aws --profile default ecr describe-repositories --repository-names maio-ml | jq -r '.repositories[0].repositoryUri'")

IMAGE_NAME := maio-ml_api_container

REGISTRY_ID := $(shell aws ecr describe-repositories --query 'repositories[?repositoryName==`'$(IMAGE_NAME)'`].registryId' --output text)

IMAGE_URI := $(REGISTRY_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

parameters:
	@echo "AWS_REGION: ${AWS_REGION}"
	@echo "IMAGE_TAG: ${IMAGE_TAG}"
	@echo "ECR_NAME: ${ECR_NAME}"
	@echo "IMAGE_NAME: ${IMAGE_NAME}"
	@echo "REGISTRY_ID: ${REGISTRY_ID}"
	@echo "IMAGE_URI: ${IMAGE_URI}"

deploy:
	@echo "Login to the ECR repository"
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(IMAGE_URI)

	@echo "Building and pushing the container"
	@docker build -t $(IMAGE_URI)/$(IMAGE_NAME) . && docker push $(IMAGE_URI)/$(IMAGE_NAME):$(IMAGE_TAG)
