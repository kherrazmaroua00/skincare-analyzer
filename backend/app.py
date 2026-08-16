from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import ast
import numpy as np

app = FastAPI(title="Skincare Ingredient Analyzer API")

KNOWN_COMMA_INGREDIENTS = ['1,2-Hexanediol', '1,3-Propanediol', '2,3-Butanediol', '1,10-Decanediol']

def smart_split_ingredients(text):
    protected = text
    for ing in KNOWN_COMMA_INGREDIENTS:
        protected = protected.replace(ing, ing.replace(',', '§'))
    parts = [p.strip().replace('§', ',') for p in protected.split(',') if p.strip()]
    return parts

# --- Load models and encoders once, at startup ---
skintype_model = joblib.load('../models/skintype_model.pkl')
benefit_model = joblib.load('../models/benefit_model.pkl')
mlb_ingredients = joblib.load('../models/ingredients_encoder.pkl')
mlb_skintype = joblib.load('../models/skintype_encoder.pkl')
mlb_benefit = joblib.load('../models/benefit_encoder.pkl')

# --- Load ingredient reference dictionary ---
def safe_parse(val):
    if isinstance(val, list):
        return val
    if pd.isnull(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

df1 = pd.read_csv('../data/processed/ingredients_reference_clean.csv')
df1['good_for_list'] = df1['good_for_list'].apply(safe_parse)
df1['avoid_list'] = df1['avoid_list'].apply(safe_parse)

df1_lookup = []
for _, row in df1.iterrows():
    if pd.isnull(row['name']):
        continue
    df1_lookup.append({
        'name_lower': row['name'].lower().strip(),
        'name': row['name'],
        'good_for': row['good_for_list'],
        'avoid': row['avoid_list']
    })

THRESHOLD_SKINTYPE = 0.45
THRESHOLD_BENEFIT = 0.35  

class IngredientsInput(BaseModel):
    ingredients: str

@app.get("/")
def root():
    return {"message": "Skincare Analyzer API is running"}

@app.post("/analyze")
def analyze(input: IngredientsInput):
    # Split raw ingredient text into a list, same way as training
    ingredient_list = smart_split_ingredients(input.ingredients)    
    # Encode ingredients into the same 405-feature format the models expect
    X_input = mlb_ingredients.transform([[i.lower().strip() for i in ingredient_list]])
    
    # Predict skin type
    proba_st = skintype_model.predict_proba(X_input)
    proba_st_matrix = np.array([p[:, 1] for p in proba_st]).T
    print("Skin type probabilities:", dict(zip(mlb_skintype.classes_, proba_st_matrix[0])))
    pred_st = (proba_st_matrix >= THRESHOLD_SKINTYPE).astype(int)
    skin_types = mlb_skintype.inverse_transform(pred_st)[0]
    
    # Predict benefits
    proba_bn = benefit_model.predict_proba(X_input)
    proba_bn_matrix = np.array([p[:, 1] for p in proba_bn]).T
    pred_bn = (proba_bn_matrix >= THRESHOLD_BENEFIT).astype(int)
    benefits = mlb_benefit.inverse_transform(pred_bn)[0]
    
    # Match against ingredient reference dictionary
    matched_actives = []
    irritant_flags = []
    for ing in ingredient_list:
        ing_lower = ing.lower().strip()
        for ref in df1_lookup:
            if ref['name_lower'] in ing_lower:
                if ref['good_for']:
                    matched_actives.append({'ingredient': ref['name'], 'good_for': ref['good_for']})
                if ref['avoid']:
                    irritant_flags.append({'ingredient': ref['name'], 'avoid_reasons': ref['avoid']})
                break
    
    return {
        "skin_type": list(skin_types),
        "benefits": list(benefits),
        "matched_actives": matched_actives,
        "irritant_flags": irritant_flags
    }

