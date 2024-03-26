from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv('../dataset/heart_clean.csv')

le=LabelEncoder()

lst=['sex','chest_pain_type','fasting_blood_sugar','resting_electrocardiogram','exercise_induced_angina','st_slope','thalassemia','disease']

label_mapping = {}

for i in lst:
    data[i]=le.fit_transform(data[i])
    label_mapping[i] = dict(zip(le.classes_, range(len(le.classes_))))

for feature, mapping in label_mapping.items():
    print(f"{feature} mapping:")
    for label_str, number in mapping.items():
        print(f"  {label_str}: {number}")

data.to_csv('heart_clean_encode.csv', index=False)