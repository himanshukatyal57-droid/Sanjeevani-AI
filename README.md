# Sanjeevani AI

A B.Tech academic project: an automated first responder and medical intake assistant that combines classical AI with machine learning.

You describe your symptoms in plain English. Rule-based triage checks for emergencies, a Decision Tree suggests possible conditions, and A* search finds the fastest hospital offering the care you need.

> **Academic demonstration only.** This is not a medical device and has not been clinically validated. In a real emergency, call 112.

## The pipeline

| Stage | What happens | Technique | Code |
|---|---|---|---|
| 1. Input | The user describes symptoms in plain English | Web form | `Frontend/` |
| 2. Symptom extraction | Finds symptoms and handles negation ("no fever") | NLP: stemming, phrase matching | `nlp_extractor.py` |
| 3. Emergency triage | IF-THEN rules assign a level, and an emergency skips stage 4 | Forward chaining | `triage_engine.py` |
| 4. Disease prediction | Suggests possible conditions, with the exact splits behind them | Decision Tree | `train_model.py`, `tree_explainer.py` |
| 5. Hospital routing | Finds the fastest hospital with the required care | A* search, built from scratch | `routing_engine.py` |
| 6. Dashboard | Shows the results, an animated route map and Debug Mode | HTML, CSS, JavaScript, SVG | `Frontend/` |

The browser talks to an Express server (`backend/`), which calls a Python API (`app.py`) that runs stages 2–5.

## Run it

You need **Python 3.10–3.13** (tested on 3.12) and **Node.js 22 or newer** (tested on 26). The dataset isn't needed, because the trained model is in the repository.

**1. Install, once:**

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm install
```

On macOS or Linux, use `python3.12 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.

`npm run start:ai` always uses this `.venv`, so a newer Python on your computer can't break it. `requirements.txt` pins the exact versions the saved model needs, which don't support Python 3.14 or newer.

**2. Start both servers, each in its own terminal:**

```
npm run start:ai
```

```
npm start
```

**3. Open <http://localhost:5000>.**

The model needs about 1 GB of free RAM and takes a few seconds to load. The card at the top of the home page shows whether each service is ready.

## Try it

Choose a duration for each one. Turn on **Academic Debug Mode** at the top of the dashboard to see the reasoning.

| Type this | Area | What it shows |
|---|---|---|
| I have crushing chest pain and I cant breathe | Lajpat Nagar | Emergency, model skipped, and a route 18 minutes faster than the closer cardiac hospital |
| I have a fever, a dry cough and a sore throat | Saket | Routine result with predictions, and the Decision Tree path in Debug Mode |
| My speech has been slurred | Karol Bagh | `EMERGENCY` for "1–3 days", but `URGENT` for "More than a week" |
| no fever but I have a headache | Any | Negation: fever is recorded as denied |
| hello there, how are you | Any | `UNKNOWN`: nothing was recognized, which isn't treated as an all-clear |

## Results

From `models/sanjeevani_model_metadata.json`, measured on a held-out test set. The data has 246,945 rows, 377 symptoms and 773 diseases.

| Symptoms given | Correct first | In top 3 |
|---|---|---|
| All (about 5) | 79% | 91% |
| 4 | 71% | 90% |
| 3 | 59% | 83% |
| 2 | 37% | 64% |

Macro-F1 is 0.79 and accuracy is 79% with all symptoms given. People often describe fewer symptoms than the dataset rows contain, so the tree is also trained on copies of rows with symptoms removed.

Across every starting point and care type on the map (210 combinations), A* matched an independent Dijkstra check every time, and explored about 18% fewer locations than Dijkstra.

## Known limitations

- **Triage rules** are illustrative and not clinically validated. They lean towards false alarms over missed emergencies, but they can still miss some.
- **Symptom extraction** uses a fixed English vocabulary, so unusual wording, Hindi and Hinglish can be missed. It can't tell who has a symptom, so "my father had a stroke" still flags a stroke, which could be happening now. A stated past time, like "in 2010" or "last year", does rule a symptom out.
- **Choosing "More than a week"** lowers speech and one-sided weakness signs from `EMERGENCY` to `URGENT`, unless the text says something like "suddenly". A real stroke reported that way would be `URGENT`.
- **Predictions** with only a few symptoms are often uncertain. Scores are model confidence, not medical certainty. Some diseases have very little data, and 19 have only one row.
- **The map** uses real Delhi NCR area names with approximate coordinates, but road distances and traffic are estimates, and all hospitals are fictional. Lines show connections, not road shapes.

## Tests

```
.venv\Scripts\python -m unittest
npm test
```

## Retraining (optional)

This needs the 190 MB Kaggle dataset, which is too large for GitHub:

1. Put `Final_Augmented_dataset_Diseases_and_Symptoms.csv` in `data/`.
2. Run `python eda_and_schema.py` to create the cleaned dataset.
3. Run `python train_model.py`, which overwrites the model in `models/`.

If you upgrade scikit-learn or NumPy, retrain and update the pinned versions in `requirements.txt`.

## Configuration

Both settings are optional and can go in a `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Website port |
| `PYTHON_API_URL` | `http://127.0.0.1:8000` | Where Express finds the Python API |
