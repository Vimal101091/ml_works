"""
Titanic Survival - Simple Model Selection Pipeline
Tests multiple models and parameters, picks the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ============== 1. LOAD DATA ==============
print("="*50)
print("TITANIC MODEL SELECTION PIPELINE")
print("="*50)

train = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============== 2. FEATURE ENGINEERING ==============
def prepare(df):
    d = df.copy()
    
    # Title
    d['Title'] = d['Name'].str.split(', ').str[1].str.split('. ').str[0]
    tm = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Dr':5,'Rev':5,'Col':5,'Major':5,'Capt':5,'Mlle':2,'Ms':2,'Mme':3}
    d['Title'] = d['Title'].map(tm).fillna(5)
    
    # Family
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['IsAlone'] = (d['FamilySize'] == 1).astype(int)
    
    # Cabin
    d['HasCabin'] = d['Cabin'].notna().astype(int)
    d['Deck'] = d['Cabin'].fillna('U').str[0]
    
    # Age
    d['Age'] = d['Age'].fillna(d.groupby(['Pclass','Sex'])['Age'].transform('median'))
    d['Age'] = d['Age'].fillna(d['Age'].median())
    
    # Fare
    d['Fare'] = d['Fare'].fillna(d.groupby('Pclass')['Fare'].transform('median'))
    d['Fare'] = d['Fare'].fillna(d['Fare'].median())
    
    # Embarked
    d['Embarked'] = d['Embarked'].fillna(d['Embarked'].mode()[0])
    
    # Encode
    d['Sex_enc'] = (d['Sex'] == 'male').astype(int)
    d['Embarked_enc'] = LabelEncoder().fit_transform(d['Embarked'])
    d['Deck_enc'] = LabelEncoder().fit_transform(d['Deck'])
    
    # Key features
    d['IsChild'] = (d['Age'] < 16).astype(int)
    d['IsWomanOrChild'] = ((d['Sex'] == 'female') | (d['Age'] < 16)).astype(int)
    d['ClassSex'] = d['Pclass'] * 10 + d['Sex_enc']
    
    feats = ['Pclass','Sex_enc','Age','SibSp','Parch','Fare','Embarked_enc',
            'Title','FamilySize','IsAlone','HasCabin','Deck_enc',
            'IsChild','IsWomanOrChild','ClassSex']
    
    return d[feats].astype(float), d['Survived']

X_train, y_train = prepare(train)
X_test, _ = prepare(test)
test_id = test['PassengerId']

print(f"Features: {X_train.shape[1]}")
print(f"Survival rate: {y_train.mean()*100:.1f}%")

# ============== 3. MODEL SELECTION ==============
print("\n" + "="*50)
print("TESTING MODELS...")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models to test
models = {
    # Random Forest variations
    'RF_100': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'RF_200': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42, n_jobs=-1),
    
    # Extra Trees
    'ET_200': ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    
    # Gradient Boosting variations
    'GB_100': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'GB_150': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    
    # Logistic Regression
    'LR': LogisticRegression(max_iter=1000, random_state=42),
}

# Test each model
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean()*100:.2f}% (+/-{scores.std()*2*100:.2f}%)")

# ============== 4. SELECT BEST MODEL ==============
print("\n" + "="*50)
print("RESULTS - SORTED BY ACCURACY")
print("="*50)

# Sort by score
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for rank, (name, score) in enumerate(sorted_results, 1):
    print(f"{rank}. {name}: {score*100:.2f}%")

best_model_name = sorted_results[0][0]
best_score = sorted_results[0][1]

print(f"\n*** BEST MODEL: {best_model_name} ({best_score*100:.2f}%) ***")

# ============== 5. TRAIN BEST & PREDICT ==============
# Train best model
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Predict
predictions = best_model.predict(X_test)

# ============== 6. SAVE SUBMISSION ==============
submission = pd.DataFrame({
    'PassengerId': test_id,
    'Survived': predictions.astype(int)
})

submission.to_csv('submission.csv', index=False)

print(f"\nPredictions: Survived = {predictions.sum()}, Not = {len(predictions)-predictions.sum()}")
print("Saved: submission.csv")
print("="*50)
print("DONE!")
print("="*50)