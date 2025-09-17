#!/bin/bash

# EKS CloudWatch Collector - Docker Run Script
# This script builds and runs the EKS collector in a Docker container

set -e

# Configuration
IMAGE_NAME="eks-cloudwatch-collector"
OUTPUT_DIR="$(pwd)/output"
AWS_CREDENTIALS_DIR="$HOME/.aws"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}EKS CloudWatch Collector - Docker Setup${NC}"

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: $0 <cluster-name> <aws-profile> [region] [days] [output-filename]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 my-cluster prod-profile"
    echo "  $0 my-cluster dev-profile us-west-2 7 custom-output.csv"
    exit 1
fi

CLUSTER_NAME="$1"
AWS_PROFILE="$2"
REGION="${3:-us-east-1}"
DAYS="${4:-30}"
OUTPUT_FILE="${5:-eks_metrics.csv}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  AWS Profile: $AWS_PROFILE"
echo "  Region: $REGION"
echo "  Days: $DAYS"
echo "  Output File: $OUTPUT_FILE"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if AWS credentials exist
if [ ! -d "$AWS_CREDENTIALS_DIR" ]; then
    echo -e "${RED}Error: AWS credentials directory not found at $AWS_CREDENTIALS_DIR${NC}"
    echo "Please ensure AWS CLI is configured with your credentials."
    exit 1
fi

if [ ! -f "$AWS_CREDENTIALS_DIR/credentials" ]; then
    echo -e "${RED}Error: AWS credentials file not found at $AWS_CREDENTIALS_DIR/credentials${NC}"
    echo "Please ensure AWS CLI is configured with your credentials."
    exit 1
fi

if [ ! -f "$AWS_CREDENTIALS_DIR/config" ]; then
    echo -e "${YELLOW}Warning: AWS config file not found at $AWS_CREDENTIALS_DIR/config${NC}"
    echo "This might be okay if you only use credentials file."
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build --load -t "$IMAGE_NAME" . > /dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker image built successfully!${NC}"
else
    echo -e "${RED}Failed to build Docker image${NC}"
    exit 1
fi

# Run the container
echo -e "${YELLOW}Running EKS CloudWatch Collector...${NC}"
docker run --rm \
    -v "$AWS_CREDENTIALS_DIR:/home/collector/.aws:ro" \
    -v "$OUTPUT_DIR:/output" \
    -e AWS_PROFILE="$AWS_PROFILE" \
    "$IMAGE_NAME" \
    --cluster-name "$CLUSTER_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$REGION" \
    --days "$DAYS" \
    --output "/output/$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Collection completed successfully!${NC}"
    echo -e "${GREEN}Output saved to: $OUTPUT_DIR/$OUTPUT_FILE${NC}"
    
    # Split CSV by resource type
    echo -e "${YELLOW}Splitting CSV by resource type...${NC}"
    SPLIT_DIR="$OUTPUT_DIR/split"
    mkdir -p "$SPLIT_DIR"
    
    python3 split_csv_by_resource.py "$OUTPUT_DIR/$OUTPUT_FILE" "$SPLIT_DIR/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}CSV files split successfully!${NC}"
        echo -e "${GREEN}Split files saved to: $SPLIT_DIR/${NC}"
    else
        echo -e "${YELLOW}Warning: CSV splitting failed, but main collection succeeded${NC}"
    fi
else
    echo -e "${RED}Collection failed${NC}"
    exit 1
fi
