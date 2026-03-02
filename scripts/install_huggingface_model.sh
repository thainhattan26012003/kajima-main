set -e

usage() {
    echo "Usage: $0 -d <path to install directory> -m <path to model directory>"
    echo "Example: $0 -d /home/user/vit_msn -m facebook/vit_msn_model"
    exit 1
}

while getopts d:m: flag
do
    case "${flag}" in
        d) HUGGINGFACE_MODEL_PATH=${OPTARG};;
        m) INSTALL_DIRECTORY=${OPTARG};;
    esac
done

if [ -z "$HUGGINGFACE_MODEL_PATH" ] || [ -z "$INSTALL_DIRECTORY" ]; then
    usage
fi

if [ -d "$INSTALL_DIRECTORY" ]; then
    echo "Removing existing $INSTALL_DIRECTORY (re-download)."
    rm -rf "$INSTALL_DIRECTORY"
fi

mkdir -p "$INSTALL_DIRECTORY"

# Download the model
echo "Downloading the model ${HUGGINGFACE_MODEL_PATH} to ${INSTALL_DIRECTORY} from Hugging Face Hub"
pip show transformers
python -c "
from transformers import AutoModel, AutoImageProcessor

try:
    print(f'Downloading the model from ${HUGGINGFACE_MODEL_PATH}')
    model = AutoModel.from_pretrained('${HUGGINGFACE_MODEL_PATH}')
    processor = AutoImageProcessor.from_pretrained('${HUGGINGFACE_MODEL_PATH}')
    model.save_pretrained('${INSTALL_DIRECTORY}')
    processor.save_pretrained('${INSTALL_DIRECTORY}')
except Exception as e:
    print(f'Error downloading the model: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Error downloading the model. Please check the model path and try again."
    exit 1
else
    echo "Model downloaded successfully to ${INSTALL_DIRECTORY}"
fi