# Kajima Construction

## Project file structure

```bash
├── Makefile
├── README.md
├── aws.env
├── checkpoints
│   ├── dinov2-base
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   ├── dinov2-base-finetuned-20250316
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── finetune_config.json
│   └── dinov2-base-finetuned-20250324
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
|       └── finetune_config.json
├── config
│   └── classification
│       ├── no_tuning
│       │   └── vit_msn_metric_predictor.json
│       └── tuning
│           ├── dinov2_base_peft_vit.json
│           ├── dinov2_base_peft_vit_20250324.json
│           └── sweep_config.json
├── docker
│   ├── app_api
│   │   └── Dockerfile
│   └── batch_processor_api
│       └── Dockerfile
├── docker-compose.yml
├── eda
│   ├── dataset_eda.ipynb
│   └── particle_analysis.ipynb
├── libs
│   ├── aws_s3.py
│   └── cybozu.py
├── media
├── openapi.json
├── poetry.lock
├── pyproject.toml
├── research
├── scripts
│   └── install_huggingface_model.sh
├── src
│   ├── __init__.py
│   ├── agent_sweep.py
│   ├── app.py
│   ├── batch_processor.py
│   ├── config.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── samplers.py
│   │   └── transforms.py
│   ├── eval.py
│   ├── main.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   ├── metric_predictor.py
│   │   ├── model_factory.py
│   │   ├── peft_vit_classifier.py
│   │   ├── utils.py
│   │   └── vit_classifier.py
│   ├── auth
│   │   ├── __init__.py
│   │   ├── cookie.py
│   │   └── jwt.py
│   ├── plot
│   ├── schema.py
│   ├── server.py
│   ├── train.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── datetime_utils.py
│   │   ├── evaluator.py
│   │   ├── losses.py
│   │   ├── meter.py
│   │   ├── optimizers.py
│   │   └── trainer.py
│   └── vars.py
└── test
    └── deployment
        └── awslocal.sh
```

### ルートファイル

- **Makefile**: プロジェクトのビルドやクリーンアップなどのタスクを自動化するためのコマンドが含まれています。
- **README.md**: プロジェクトのドキュメントです。
- **aws.env**: AWS 設定用の環境変数です。
- **docker-compose.yml**: マルチコンテナアプリケーションを管理するための Docker Compose の設定です。
- **openapi.json**: プロジェクトの API の OpenAPI 仕様です。
- **poetry.lock** と **pyproject.toml**: Python プロジェクトの依存関係管理ファイルです。

### 主要ディレクトリ

- **checkpoints/**: 事前学習済みおよびファインチューニング済みのモデルファイルが格納されます。
  - `dinov2-base/`: ベースモデルファイルです。
  - `dinov2-base-finetuned-*`: 設定ファイルとアダプターファイルを含むファインチューニング済みのモデルファイルです。
- **config/**: モデルの学習とチューニング用の設定ファイル。
  - `classification/`: 分類タスク用のJSONファイルが格納されています。
- **docker/**: コンテナ化されたアプリケーションを構築するためのDockerfiles。
  - `app_api/`: WebアプリケーションAPI用のDockerfiles。
  - `batch_processor_api/`: バッチ処理API用のDockerfiles。
- **eda/**: 探索的データ分析ノートブック。
- **libs/**: 外部サービス用のユーティリティスクリプト。
  - `aws_s3.py`: AWS S3関連ユーティリティ。
  - `cybozu.py`: Cybozu関連ユーティリティ。
- **media/**: ドキュメントで使用する画像やメディアファイルが格納されています。
- **research/**: 研究関連のファイルや実験用のプレースホルダ。
- **scripts/**: 自動化用のシェルスクリプト。
  - `install_huggingface_model.sh`: Hugging Face モデルをインストールするためのスクリプト。
- **src/**: プロジェクトのメインソースコード。
  - `app.py`: ウェブアプリケーションのエントリポイント。
  - `train.py`: モデルをトレーニングするためのスクリプト。
  - `eval.py`: モデルを評価するためのスクリプト。
  - `data/`: データ処理ユーティリティ。
  - `model/`: モデル関連のユーティリティと実装。
  - `utils/`: 一般的なユーティリティスクリプト。
  - `auth/`: 認証関連のユーティリティ。
- **test/**: テスト関連のファイル。
  - `deployment/`: デプロイメントテストスクリプト。
    - `awslocal.sh`: AWS のサービスをローカルでテストするためのスクリプト。

## Setup

1. Install poetry

```bash
pip install poetry
```

2. Install dependencies

```bash
poetry self update
poetry self add poetry-plugin-export
poetry install
```

## Usage

Create `.env` file

```bash
cp .env.example .env
```

Fill the path to configuration file `MODEL_CONFIG_PATH` variable and run this command to export the environment variables

```bash
source .env
```

Running the Web App

```bash
poetry run python -m src.app
```

## Webアプリ

1. ブラウザで以下にアクセス
<http://127.0.0.1:7860/>

2. 画像をアップロード

![alt text](media/image.jpg)

3. 推論結果が画面右側に表示される (7_{1~4}が表示される)

![alt text](media/image-1.png)

## Training model

1. データセットを用意する

次の形式でトレーニング データセットとテスト データセットを作成します。

```
kajima_train_dataset
├── 20241212 <-- date
│   ├── 7-1-241212 <--- Format: random_number-class_label-date
│   │   ├── IMG_6154.JPG <---- Image file
│   │   ├── IMG_6155.JPG
│   │   ├── IMG_6156.JPG
│   │   ├── IMG_6157.JPG
 ........
│   ├── 7-2-241212
│   │   ├── IMG_6267.JPG
│   │   ├── IMG_6268.JPG
│   │   ├── IMG_6269.JPG
 ........
│   ├── 7-3-241212
│   │   ├── 7-3全景.JPG
│   │   ├── 7-3地山.JPG
│   │   ├── 7-3ほぐし.JPG
 .........
│   ├── 7-4-241212
│   │   ├── IMG_6110.JPG
│   │   ├── IMG_6111.JPG
│   │   ├── IMG_6118.JPG
│   │   ├── IMG_6119.JPG
│   │   ├── IMG_6120.JPG
 .........
├── 20250108
│   ├── 7-1-250108
│   ├── 7-2-250108
│   ├── 7-4-250108
 ..........
```

2. 設定ファイルを作成する

次の形式で JSON 設定ファイルを作成します。`config/` ディレクトリにあるサンプル設定ファイルを参照できます。

#### JSON Configuration structure

`model`: Model configuration (required)

`data`: Dataset configuration (required in case of training)

`training`: Training configuration

`testing`: Testing configuration

`inference`: Inference configuration

#### Configuration Options

**Model Configuration (Required)**

```json
"model": {
  "method": "tuning",           // "tuning" or "no_tuning" (Required)
  "model_name": "my_model",     // Name of your model (Required)
  "model_path": "/path/to/model", // Path to model file (Required)
  "feature_matrix_path": null,  // Required if method is no_tuning and in case of eval
  "peft_adapter_path": null,    // Required if method is tuning and in case of eval (Not required in case of training)
  "classifier_head": "cosine"   // Required if method is tuning. Currently support: ["cosine", "linear", "layernorm", or "l2norm"]
}
```

**Dataset Configuration (Required in case of training)**

```json
"data": {
  "train_dataset_dir": "/path/to/dataset", // Path to training dataset
  "test_dataset_dir": "train_dataset",    // Path to testing dataset (Required in case of testing)
  "val_dataset_dir": "val_dataset",        // Path to validation dataset (Optional)
  "num_classes": 4       // Number of classes (Required)
}
```

**Training Configuration**

Configuration for method = "tuning"

```json
"training": {
  "batch_size": 8,
  "criterion_type": "CBL",                    // "CBL", "LA", "focal", "CE", "LADE"
  "init_head": "class_mean",                  // "class_mean" or "no"
  "lora_initialization_strategy": "dora",     // "dora", "random", "rslora"
  "lora_rank": 16,
  "lr": 1e-6,
  "micro_batch_size": 8, // Have to be smaller than or equal to batch_size
  "num_epochs": 10,
  "num_layer_finetuned": null,                // null for all layers, or specific number
  "optimizer": "adam",                        // "adam" or "lion"
  "sampler": "class_aware_sampler"            // "down_sampler", "class_aware_sampler", or ""
}
```

Configuration for method = "no_tuning"

```json
"training": {
  "batch_size": 8,
  "num_workers": 0
}
```

**Testing Configuration**

```json
"testing": {
  "batch_size": 2,
  "num_workers": 0
}
```

**Inference Configuration**

*Important:* `resolution` と `crop_size` をモデルの入力サイズと同じ値に設定する必要があります。

```json
"inference": {
  "resolution": 224,
  "crop_size": 224
}
```

3. 学習を実行する

#### Command line usage

```bash
poetry run python -m src.train -c /path/to/config.json -o /path/to/output/directory
```

Command line arguments:

`-c, --configuration`: 設定ファイルへのパス（必須）

`-o, --output-dir`: モデル出力ディレクトリへのパス

`--device`: トレーニングに使用するデバイス (デフォルトでは、マシン上の GPU を自動的に検出します。存在しない場合は、CPU を使用します)

#### Examples

For method = "no_tuning"

```bash
poetry run python -m src.train -c config/classification/no_tuning/vit_msn_metric_predictor.json -o checkpoints/vit-msn-small --device cpu
```

For method = "tuning"

```bash
poetry run python -m src.train -c /Users/lamle/Desktop/kajima-construction/config/classification/tuning/dinov2_base_peft_vit_20250324.json -o checkpoints --device cpu
```

## モデルを評価する

#### Command line usage

```bash
poetry run python -m src.eval -c path/to/config.json -d device_to_use
```

Command line arguments:

`-c, --configuration`: 設定ファイルへのパス（必須）

`-d, --device`: 評価に使用するデバイス (デフォルトでは、マシン上の GPU を自動的に検出します。存在しない場合は、CPU を使用します)

#### 例

For method = "no_tuning"

```bash
poetry run python -m src.eval -c config/classification/no_tuning/vit_msn_metric_predictor.json -d cpu
```

For method = "tuning"

```bash
poetry run python -m src.eval -c config/classification/tuning/dinov2_base_peft_vit_20250324.json -d cpu
```

#### Output

![evaluation output](media/eval_output.jpg)

## ハイパーパラメータ調整のための WanDB のエージェントスイープ

モデルのハイパーパラメータを自動的に調整するために、`wandb` スイープ エージェントを使用することもできます。次のコマンドは、異なるハイパーパラメータで 10 回の試行を実行するスイープ エージェントを起動します。

スイープ構成ファイルは `config/classification/tuning/sweep_config.json` にあります。

1. wandbアカウントにログイン

```bash
wandb login
```

2. スイープエージェントを起動する

#### Command line usage

```bash
poetry run python -m src.agent_sweep -c config/classification/tuning/sweep_config.json --sweep-count 3 -b checkpoints/dinov2-base -o temps --project-name soil-classification \
--train-dataset-dir datasets/kajima_dataset \
--test-dataset-dir datasets/kajima_test_dataset
```

#### Command line arguments

`-c, --configuration`: 設定ファイルへのパス（必須）

`--sweep-count`: 実行する試行回数（デフォルト: 10）

`-b, --base-model-path`: トレーニングに使用するベースモデル (デフォルト: dinov2-base)

`-o, --output-dir`: モデル出力ディレクトリへのパス

`--project-name`: wandbプロジェクトの名前

`--wandb-entity`: wandbエンティティの名前

`--train-dataset-dir`: トレーニングデータセットへのパス

`--test-dataset-dir`: テストデータセットへのパス

`--device`: トレーニングに使用するデバイス (デフォルトでは、マシン上の GPU を自動的に検出します。存在しない場合は、CPU を使用します)

#### Output example

![wandb sweep output](media/wandb_sweep_output.jpg)

## Deployment

aws cli を認証する

```bash
aws configure
```

ルート ディレクトリに次の内容の `aws.env` ファイルを作成します。

```bash
AWS_REGION="your-aws-region"
AWS_ACCOUNT_ID="your-aws-account-id"
AWS_DOMAIN="amazonaws.com"
AppApiRepositoryName="name-of-your-app-api-repository"
BatchProcessorRepositoryName="name-of-your-batch-processor-repository"
```

次に、以下のコマンドを実行して環境変数をエクスポートします。

```bash
source aws.env
```

### API Deployment

このセクションでは、AWS Lambda関数にAPIサーバーをデプロイする方法について説明します。

#### 1. Build Docker image

次のコマンドを実行して、API Docker イメージをビルドします。

```bash
make build-app-api
```

#### 2. API Docker イメージを API ECR リポジトリにプッシュする

```bash
make authenticate
make push-app-api
```

#### 3. AWS Lambda関数に新しいAPI Dockerイメージをデプロイする

*重要*:

- AWS Lambda 関数をすでに作成している場合にのみ、API Docker イメージを AWS Lambda 関数にデプロイできます。

```bash
API_LAMBDA_FUNCTION_NAME="your-api-lambda-function-name"
IMAGE_URI="$(aws ecr describe-repositories --repository-names $AppApiRepositoryName --query 'repositories[0].repositoryUri' --output text):latest"

aws lambda update-function-code \
    --function-name $API_LAMBDA_FUNCTION_NAME \
    --image-uri $IMAGE_URI
```

### Model Deployment

このセクションでは、AWS Lambda 関数でモデルをデプロイする方法について説明します。

現在、デプロイプロセスは method="tuning" が指定されたモデルのみをサポートしています。

#### 1. 設定ファイルを準備する

トレーニング後は、次のファイルが作成されます。

1. 設定ファイル (Example: `config/classification/tuning/dinov2_base_peft_vit_20250324.json`)
2. ベースモデルフォルダ (Example: `checkpoints/dinov2-base`)
3. Peftアダプタファイル (Example: `checkpoints/dinov2-base-finetuned-20250324`)

#### 2. Copy those files to Docker image

上記の設定ファイルをルートディレクトリの `config` フォルダにコピーします。

ベースモデルフォルダと peft アダプタフォルダをルートディレクトリの `checkpoints` フォルダにコピーします。

*注*:

- 設定ファイル内のモデルと peft アダプタのパスが正しいことを確認してください。パスは `checkpoints/your-model-folder` と `checkpoints/your-peft-adapter-folder` である必要があります。
- デプロイするモデルフォルダと peft アダプタフォルダのみを残してください。その他のフォルダは削除しても構いません。これにより、Docker イメージのサイズを縮小できます。

#### 3. Build Docker image

次に、次のコマンドを実行して Docker イメージをビルドします。

```bash
make build-batch-processor-api
```

#### 4. Push Docker image to ECR

次のコマンドを実行して、Docker イメージを ECR にプッシュします。

```bash
make authenticate
make push-batch-processor-api
```

#### 5. Deploy the new model in AWS Lambda function

*重要*:

- AWS Lambda 関数をすでに作成している場合にのみ、API Docker イメージを AWS Lambda 関数にデプロイできます。


`.env` ファイルを作成する

```bash
cp .env.example .env
```

`.env` ファイルに情報を入力します。内容は、デプロイメントマニュアルの `secrets.tfvars` 内の `app_env` オブジェクトと同じです。

*注*: `MODEL_CONFIG_PATH` 変数は、新しいモデル構成ファイルへのパスである必要があります。


```bash
BATCH_PROCESSOR_LAMBDA_FUNCTION_NAME="your-batch-processor-lambda-function-name"
IMAGE_URI="$(aws ecr describe-repositories --repository-names $BatchProcessorRepositoryName --query 'repositories[0].repositoryUri' --output text):latest"

aws lambda update-function-code \
    --function-name $BATCH_PROCESSOR_LAMBDA_FUNCTION_NAME \
    --image-uri $IMAGE_URI
```
