# Engine Predictive Maintenance

Capstone Project 3: predicts whether an engine needs maintenance from 6 sensor readings
(engine rpm, lub oil pressure, fuel pressure, coolant pressure, lub oil temperature, coolant temperature).

## Folder structure

```
├── .github/workflows/pipeline.yml   GitHub Actions pipeline (registration -> preparation -> training)
├── data/engine_data.csv             dataset (provided with the capstone brief)
├── src/register_data.py             step 1: checks the dataset and saves data/registry.json
├── src/prep.py                      step 2: cleans the data and saves the train/test split
├── src/train.py                     step 3: tunes 6 models, logs them in MLflow, saves the best model
├── models/                          best model, its metadata and results (committed by the pipeline)
├── app.py                           Streamlit app that loads models/best_model.joblib
├── requirements.txt                 libraries for the Streamlit app
├── requirements-pipeline.txt        libraries for the pipeline
└── Rajendran_PredictiveMaintenance_Notebook.ipynb   notebook with EDA and all steps
```

## Run locally

```
pip install -r requirements-pipeline.txt -r requirements.txt
python src/register_data.py
python src/prep.py
python src/train.py
streamlit run app.py
```

## Pipeline

Every push to `main` that changes `data/`, `src/`, `requirements-pipeline.txt` or the workflow runs the pipeline.
It can also be started from the Actions tab (Run workflow). The data preparation job passes the train/test split
to the training job as the artifact `train-test-data`, and the training job commits the best model back to `main`.
