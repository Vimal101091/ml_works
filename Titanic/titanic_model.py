import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION MODEL")
print("=" * 60)

train = pd.read_csv('E:/MOOC/Udemy Courses/AI&ML/Kaggle/train.csv')
test = pd.read_csv('E:/MOOC/Udemy Courses/AI&ML/Kaggle/test.csv')

print(f"\n[1] DATA OVERVIEW")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nSurvival rate: {train['Survived'].mean()*100:.2f}%")

print(f"\n[2] MISSING VALUES")
print(f"Train - Age: {train['Age'].isna().sum()}, Cabin: {train['Cabin'].isna().sum()}, Embarked: {train['Embarked'].isna().sum()}")
print(f"Test - Age: {test['Age'].isna().sum()}, Fare: {test['Fare'].isna().sum()}, Cabin: {test['Cabin'].isna().sum()}")

def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

def get_deck(cabin):
    if pd.isna(cabin):
        return 'U'
    return cabin[0]

def process_data(df, is_train=True):
    df = df.copy()
    
    df['Title'] = df['Name'].apply(extract_title)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].apply(lambda x: title_mapping.get(x, 5) if x in title_mapping else 5)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].apply(get_deck)
    
    age_median = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    if is_train:
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    else:
        df['Embarked'] = df['Embarked'].fillna('S')
    
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median')[df.index[0]:].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df['Sex'] = (df['Sex'] == 'male').astype(int)
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'].fillna('S'))
    df['Deck'] = LabelEncoder().fit_transform(df['Deck'])
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
              'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'Deck']
    
    X = df[features]
    
    if is_train:
        y = df['Survived']
        return X, y
    return X

X_train, y_train = process_data(train, is_train=True)
X_test, process_data(test, is_train=False)

test_passenger_id = test['PassengerId']

print(f"\n[3] FEATURES USED")
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'Deck']
for f in features:
    print(f"  - {f}")

X_test = process_data(test, is_train=False)

print(f"\n[4] MODEL TRAINING")
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross Validation Results:")
print(f"  Scores: {cv_scores}")
print(f"  Mean Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

rf.fit(X_train, y_train)

print(f"\n[5] FEATURE IMPORTANCE")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

print(f"\n[6] PREDICTION & SUBMISSION")
predictions = rf.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': predictions.astype(int)
})

submission.to_csv('E:/MOOC/Udemy Courses/AI&ML/Kaggle/submission.csv', index=False)

print(f"Survived: {predictions.sum()}, Not Survived: {len(predictions) - predictions.sum()}")
print(f"Submission saved to: submission.csv")

print("\n" + "=" * 60)
print("MODEL COMPLETE!")
print("=" * 60)