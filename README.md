# House Price Prediction API

Small FastAPI service that loads a trained Keras model and supporting
artifacts to predict real-estate asking prices from structured inputs.

Project layout

- `app.py` — FastAPI application exposing `/predict` which expects JSON
  payloads shaped like `{ "data": { <column>: [value, ...], ... } }`.
- `train_house_price_model.py` — (training scripts, kept for reference).
- `model/` — model artifacts used by `app.py` (encoder, scaler, column lists,
  and `real_estate_model.keras`).
- `utils/validator.py` — helper functions to load allowed categorical
  values from the saved `OneHotEncoder` and validate string inputs.
- `tests.py` — simple, modular test runner that POSTs example inputs to
  the local API and reports predicted vs expected prices.
- `requirements.txt` — Python dependencies for running & developing.

Quick start (development)

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the `model/` directory contains the trained artifacts used
   by `app.py`:

- `real_estate_model.keras` (Keras SavedModel/Weights),
- `one_hot_encoder.joblib`, `robust_scaler.joblib`, `num_cols.joblib`,
  `cat_cols.joblib`.

4. Run the API locally:

```bash
uvicorn app:app --reload --port 8000
```

5. Run the provided test runner to exercise a set of example inputs:

```bash
python3 tests.py
```

`tests.py` prints each test's predicted price, the expected (reported)
price, the absolute difference and percent difference, and a final
summary.

Taihaku CSV tests

The repository also includes `test_taihaku_csv.py`, which runs the
prediction API against MLIT transaction CSVs for Taihaku Ward and
compares predictions to the recorded sale price in each row.

Typical usage:

```bash
python3 test_taihaku_csv.py --csv "Miyagi Prefecture_Taihaku Ward_20174_20184.csv" --limit 100 --output "taihaku_2018_results.csv" > "test_taihaku_modal_2018.txt"
```

How it works:

- Reads one Taihaku CSV and converts MLIT Japanese columns into the API
  feature schema.
- Normalizes known district / station / zoning labels into the model's
  canonical category strings.
- Uses `api/utils/validator.py` before sending any request. This is
  required: the test runner should validate generated payloads against
  the encoder vocabulary instead of posting unchecked category values.
- Uses same-file station-time median imputation only when station name
  exists but station time is blank.
- Skips rows that are still missing required model inputs such as
  structure, building year, total floor area, or blank station name.
- Writes a results CSV and prints a text summary. Skipped rows include
  the skip reason plus the exact failing values, which makes mapping and
  parsing problems easier to debug.

Validator

The file `utils/validator.py` contains helpers to load allowed
categorical values from the saved `OneHotEncoder` and validate user
inputs (e.g., `NearestStation`, `CityPlanning`). The validator can
suggest close matches when an input value isn't recognized; the API
returns a helpful validation error in that case.

If you get validation errors from the API like:

```
{"error": "Input validation failed", "details": {"CityPlanning": [...]}}
```

it means one of the categorical string values you provided isn't in the
encoder's vocabulary. Either choose one of the suggested values or add a
normalization/mapping layer to translate synonyms to the canonical
category before sending requests.

Tests & CI suggestions

- The repository includes a simple runner (`tests.py`). If you'd like a
  tighter automated test, I can convert these into `pytest` tests and add
  a tolerance threshold so CI fails when predictions deviate too far
  from expected values.
- When running tests in CI, ensure the `model/` artifact files are
  available (or mock the calls) so tests are deterministic.

Notes about `requirements.txt` and Kaggle

- `requirements.txt` currently includes an unpinned `kaggle` entry. If
  your environment does not require the Kaggle client, you can remove
  or pin it. If you keep it, remember the Kaggle CLI requires a
  `kaggle.json` API token in `~/.kaggle/` with secure permissions.

Troubleshooting

- If the server raises a 500 error for inputs that omit a categorical
  column (or pass unexpected types), inspect `app.py` where the
  DataFrame is constructed and `encoder.transform` is called. The
  validator can be used before encoding to avoid these errors.
