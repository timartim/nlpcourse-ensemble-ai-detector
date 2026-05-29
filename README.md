# BERT AI Detector

Service for scoring Russian texts with a BERT-based AI-text detector.

## Structure

```text
.
├── src/
│   ├── model_service/          # Shared detector class for API, CLI and UI
│   ├── backend/                # FastAPI model service
│   ├── frontend/               # Streamlit UI
│   ├── cli.py                  # Training, testing, scoring and heatmap CLI
│   ├── data_models/            # Existing model code
│   ├── heatmap/                # Detector visualization and heatmap analysis
│   ├── scripts/                # Dataset collection and obfuscator scripts
│   └── notebooks/              # Research notebooks
├── trained_models/             # Local model artifacts, ignored by git
├── data/
│   ├── train_data/             # Local training/evaluation datasets, ignored by git
│   └── output_data/            # Generated reports, ignored by git
├── dataset_images/             # Lightweight documentation images
├── Examples/                   # Text examples
├── docker/
│   ├── backend/Dockerfile
│   └── frontend/Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-backend.txt
└── requirements-frontend.txt
```

## Run With Docker

```bash
docker compose up --build
```

Backend API:

```text
http://localhost:8000
```

Streamlit UI:

```text
http://localhost:8501
```

Default model path used by Docker Compose:

```text
trained_models/ensemble_models_2_3000/manifest.json
```

## CLI

Score one text:

```bash
PYTHONPATH=src python -m cli score \
  --detector-type ensemble \
  --model-path trained_models/ensemble_models_2_3000/manifest.json \
  --text "Текст для проверки"
```

Obfuscate one text:

```bash
PYTHONPATH=src python -m cli obfuscate \
  --detector-type ensemble \
  --model-path trained_models/ensemble_models_2_3000/manifest.json \
  --rewrite-threshold 0.8 \
  --text "Данная система является примером текста для проверки."
```

Obfuscate a dataset:

```bash
PYTHONPATH=src python -m cli obfuscate \
  --detector-type ensemble \
  --model-path trained_models/ensemble_models_2_3000/manifest.json \
  --dataset data/train_data/articles.jsonl \
  --text-col text \
  --output-path data/output_data/articles_obfuscated.jsonl \
  --log-path data/output_data/articles_obfuscation_log.jsonl
```

Evaluate on a labeled dataset:

```bash
PYTHONPATH=src python -m cli test \
  --detector-type hf \
  --model-path trained_models/Models/epoch_03 \
  --dataset data/train_data/final_dataset.parquet \
  --text-col text \
  --label-col label
```

Build heatmap artifacts:

```bash
PYTHONPATH=src python -m cli heatmap \
  --detector-type ensemble \
  --model-path trained_models/ensemble_models_2_3000/manifest.json \
  --input-path data/train_data/Ru-hard-detection-dataset-main/long_sc/paraphrased_generated_articles.json \
  --output-dir data/output_data/scientific_heatmap_run \
  --text-col text \
  --article-id-col article_id \
  --threshold 0.8
```

Fine-tune a single HuggingFace classifier:

```bash
PYTHONPATH=src python -m cli train \
  --dataset data/train_data/final_dataset.parquet \
  --base-model DeepPavlov/rubert-base-cased \
  --output-dir trained_models/custom_hf_model \
  --text-col text \
  --label-col label
```

Compare the project detector with a pretrained Russian HuggingFace detector:

```bash
PYTHONPATH=src python src/scripts/compare_detectors.py \
  --dataset data/train_data/combined_human_labeled.csv \
  --generated-text-col generatedText \
  --generator-col llmUsed \
  --generators chatgpt5 deepseek3.2 \
  --own-detector-type ensemble \
  --own-model-path trained_models/ensemble_models_2_3000/manifest.json \
  --baseline-model orzhan/ruroberta-ruatd-binary \
  --output-predictions data/output_data/detector_comparison.jsonl \
  --output-summary data/output_data/detector_comparison_summary.json
```

## API

Health check:

```bash
curl http://localhost:8000/health
```

Score text:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"texts":["Текст для проверки"],"detector_type":"ensemble","model_path":"trained_models/ensemble_models_2_3000/manifest.json"}'
```

Obfuscate text:

```bash
curl -X POST http://localhost:8000/obfuscate \
  -H "Content-Type: application/json" \
  -d '{"texts":["Данная система является примером."],"detector_type":"ensemble","model_path":"trained_models/ensemble_models_2_3000/manifest.json","rewrite_threshold":0.8}'
```
