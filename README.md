# SkinSense — Skincare Ingredient Safety & Match Analyzer

Paste a skincare product's ingredient list and get back a predicted skin-type fit, likely benefits, matched active ingredients, and flagged potential irritants — powered by machine learning models trained on real product data.

**Live demo:** [skincare-analyzer-eight.vercel.app]

---

## Overview

Most people can't easily interpret a skincare ingredient list — cosmetic chemistry names don't mean much without expert knowledge. SkinSense analyzes any product's ingredients and returns:

- **Skin-type suitability** — which skin types (Oily, Dry, Combination, Normal) the product is likely well-suited for
- **Predicted benefits** — likely effects such as hydration, anti-aging, acne treatment, or brightening
- **Active ingredients** — known beneficial ingredients detected, with what they help with
- **Irritant flags** — known allergens or ingredients some users should avoid

The goal is to give clear, ingredient-based insight without requiring the user to research every component manually.

---

## How It Works

The system is split into three machine learning tasks plus one rule-based lookup, all driven by the product's ingredient list:

| Task | Approach | Result |
|---|---|---|
| Skin-type suitability | Multi-label Logistic Regression (`class_weight='balanced'`) | Macro F1: 0.80 |
| Benefit prediction | Multi-label Logistic Regression | Macro F1: 0.56 |
| Active ingredient / irritant flagging | Rule-based substring matching against a curated ingredient dictionary | — |

Ingredients are encoded as multi-hot features (405 ingredients that appeared in 8+ products, out of ~3,000 unique raw ingredients) rather than the full vocabulary, to keep the feature space reasonable relative to the dataset size. Both classification models use tuned probability thresholds (found via F1 sweep across cross-validation) rather than the default 0.5 cutoff.

---

## Data

- **510 real skincare products** (Sephora product catalog), with ingredient lists and skin-concern/benefit tags derived from product highlight metadata
- **247 curated ingredients** with known benefits and allergy/irritant flags, used as a reference dictionary for the matching task

### Known limitations (documented honestly, not hidden)
- Only ~194 of 510 products had explicit skin-type tags, and ~357 had benefit tags — smaller classes (e.g. Oily skin, Dark spots benefit) have less training data and correspondingly weaker, less stable predictions than well-represented classes (Combination, Normal, Dryness).
- Early model versions showed a baseline bias toward majority skin-type classes, discovered via an empty-input sanity check (predicting probabilities with zero ingredients as input). This was mitigated using `class_weight='balanced'` and threshold re-tuning; it isn't fully eliminated, since it partly reflects a genuine skew in the training data itself (most products in the dataset are tagged suitable for Combination/Normal skin).
- The ingredient reference dictionary covers 247 well-known/notable ingredients, not the full universe of possible INCI ingredients — irritant/active detection is best-effort, not exhaustive.
- Multi-item product sets/kits (~5% of the raw data) were excluded, since their ingredient lists don't correspond to a single product.

---

## Tech Stack

- **ML:** Python, scikit-learn, pandas
- **Backend:** FastAPI
- **Frontend:** HTML/CSS/JavaScript (vanilla)
- **Deployment:** Render (API), Netlify (frontend)

---

## Project Structure

```
skincare-analyzer/
├── data/
│   ├── raw/               # original source data
│   └── processed/         # cleaned datasets used for training
├── notebooks/
│   └── exploration.ipynb  # data cleaning, feature engineering, model training/evaluation
├── models/                # saved trained models and label encoders
├── backend/
│   └── app.py             # FastAPI service
├── frontend/
│   └── index.html         # analyzer UI
└── README.md
```

---

## Running Locally

**Backend:**
```bash
cd backend
pip install -r ../requirements.txt
uvicorn app:app --reload
```
API runs at `http://127.0.0.1:8000` — interactive docs at `/docs`.

**Frontend:**
Open `frontend/index.html` directly in a browser (no build step required).

---

## Example

**Input:**
```
Water, Glycerin, Niacinamide, Cetearyl Alcohol, Retinol, Ascorbic Acid, Fragrance (Parfum), Sodium Hyaluronate
```

**Output:**
```json
{
  "skin_type": ["Combination", "Normal"],
  "benefits": ["Anti-Aging", "Dryness"],
  "matched_actives": [
    {"ingredient": "Retinol", "good_for": ["Fine Lines", "Elasticity", "Wrinkles"]}
  ],
  "irritant_flags": [
    {"ingredient": "Ascorbic Acid", "avoid_reasons": ["Sensitive", "Related Allergy"]}
  ]
}
```

---

## Out of Scope (v1)

- User accounts / saved history
- Camera/OCR-based ingredient scanning (text input only)
- Personalized recommendations across a product catalog

## Possible Future Work

- Ingredient scanning via camera/OCR
- Larger, more balanced training dataset (particularly more Oily-skin and rare-benefit examples)
- Confidence scores exposed in the UI, not just binary suitability

---

## Author

Built by [Maroua Kherraz] as a portfolio project applying classic ML (scikit-learn) to a real-world, imperfect dataset — end to end, from data cleaning through a deployed web app.
