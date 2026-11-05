# Digital Declutter AI

A local dashboard that compares a base language model with a fine-tuned model on labeled digital-item descriptions. It returns suggested actions for human review; it does not connect to user accounts or delete anything.

## Run the app on Windows

### 1. Install Python dependencies

From the project folder in Command Prompt:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Python 3.11 or newer is recommended.

### 2. Configure the model provider

Copy `.env.example` to `.env`, then set the provider and exact Ollama model tags:

```cmd
copy .env.example .env
```

For local Ollama inference, make sure both models are installed (`ollama list`) and set `.env`:

```env
DECLUTTER_PROVIDER=ollama
DECLUTTER_BASE_MODEL=qwen2.5:0.5b
DECLUTTER_FINE_TUNED_MODEL=YOUR_INSTALLED_FINE_TUNED_TAG
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

Replace `YOUR_INSTALLED_FINE_TUNED_TAG` with the exact local tag shown by `ollama list`. Keep the base and fine-tuned tags distinct. Ollama must be running before you start the API.

### 3. Start the backend

In the first Command Prompt, from the project folder:

```cmd
.venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

Check that `http://127.0.0.1:8000/api/health` responds with JSON.

### 4. Start the dashboard

Open a second Command Prompt in the project folder:

```cmd
python -m http.server 5173 --directory frontend
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser. Upload a CSV or Excel file and select **Run Analysis**.

## Dataset format

Each uploaded file needs an `item_text` column. Include `actual_action` to calculate evaluation metrics. Supported labels are `KEEP`, `DELETE`, `ARCHIVE`, `REVIEW`, and `UNSUBSCRIBE`. Without `actual_action`, the app returns model recommendations and leaves metrics blank.

For evaluation, use a held-out labeled dataset that was not used for fine-tuning. The dashboard compares both models on the same rows and displays accuracy, macro precision, macro recall, macro F1, confusion matrices, and all row predictions.

## Generate domain datasets

The project keeps synthetic candidate datasets in separate folders for emails, photos, files, downloads, and subscriptions. Generate 500 records per domain:

```cmd
python scripts/generate_datasets.py --records-per-domain 500 --output-root datasets_v2
python scripts/prepare_splits.py --data-root datasets_v2 --include-synthetic
```

This produces train, validation, and test JSONL splits under each domain. Synthetic records are for development and should not be treated as evidence of real-world model performance. For reviewed examples, put approved CSV or JSONL files under each domain's `curated` folder and prepare splits without `--include-synthetic`.

## Fine-tune with Nebius Token Factory

Fine-tuning is optional; the dashboard only needs two installed/available model IDs. Add `NEBIUS_API_KEY` to the local `.env` file, then submit the prepared splits:

```cmd
python training/submit_nebius_job.py --data-root datasets_v2 --include-synthetic --model Qwen/Qwen2.5-0.5B-Instruct
```

Cloud fine-tuning and inference may incur charges. The script saves job metadata and uploaded files under `artifacts/`, which is excluded from Git. After the job succeeds, deploy or download its fine-tuned model, configure `DECLUTTER_FINE_TUNED_MODEL`, and restart the backend.

## Project structure

```text
backend/       FastAPI upload, inference, and evaluation API
frontend/      Dashboard source
datasets/      Per-domain curated dataset folders and schemas
scripts/       Dataset generation and split preparation
training/      Nebius submission and local adapter tools
evaluation/    Evaluation utilities
```

Keep `.env` and API keys private. Treat all recommendations as suggestions that require human review.
