from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

data = pd.read_csv('../dataset/heart_clean_encode.csv')

X = data.drop('disease', axis=1)
y = data['disease']

# split into training, testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {}
    for key, value in request.form.items():
        features[key] = value

    model_type = features.pop('modelType', 'logisticRegression')

    # Chọn mô hình dựa trên loại mô hình được chọn
    if model_type == 'logisticRegression':
        model = LogisticRegression()
        hyperparameters = {
            'C': float(features['C']) if features.get('C', '') else 1.0,
            'penalty': features['penalty'] if features.get('penalty', '') else 'l2',
            'solver': features['solver'] if features.get('solver', '') else 'liblinear'
        }
    elif model_type == 'decisionTree':
        model = DecisionTreeClassifier()
        hyperparameters = {
        'max_depth': int(features['max_depth']) if features.get('max_depth', '') else None,
        'min_samples_split': int(features['min_samples_split']) if features.get('min_samples_split', '') else 2,
        'min_samples_leaf': int(features['min_samples_leaf']) if features.get('min_samples_leaf', '') else 1
    }
    elif model_type == 'knn':
        model = KNeighborsClassifier()
        hyperparameters = {
            'n_neighbors': int(features['n_neighbors']) if features.get('n_neighbors', '') else 5,
            'weights': features.get('weights') if features.get('weights', '') else 'uniform',
            'algorithm': features.get('algorithm') if features.get('algorithm', '') else 'auto'
        }
    elif model_type == 'svm':
        model = SVC(probability=True)
        hyperparameters = {
            'C': float(features['C_svm']) if features.get('C_svm', '') else 1.0,
            'kernel': features.get('kernel') if features.get('kernel', '') else 'rbf',
            'gamma': features.get('gamma') if features.get('gamma', '') else 'scale'
        }
    elif model_type == 'naiveBayes':
        model = GaussianNB()
        hyperparameters = {}
    else:
        return json.dumps({'error': 'Invalid model type'}), 400

    # Sử dụng giá trị để tinh chỉnh mô hình
    model.set_params(**hyperparameters)
    comment = evaluate_model_performance(model)
    model.fit(X, y)

    # Sử dụng features để đưa ra dự đoán
    input_features = np.array([[float(features[key]) for key in ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'resting_electrocardiogram', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia']]])
    prediction = model.predict(input_features)

    return json.dumps({'prediction': int(prediction[0]), 'model': model_type, 'comment': comment})

def evaluate_model_performance(model):
    model_train = model
    model_train.fit(X_train, y_train)

    predictions = model_train.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    tn, fp, fn, tp = cm.ravel()
    npv = tn / (tn + fn)
    specificity = tn / (tn + fp)
    
    # Sinh bình luận dựa trên các độ đo hiệu suất
    accuracy_comment = f"Mô hình đạt độ chính xác {accuracy:.2%} trên toàn bộ tập kiểm thử. Điều này có nghĩa là tỷ lệ dự đoán đúng về tình trạng mắc bệnh hoặc không mắc bệnh của bệnh nhân. "
    precision_comment = f"Tỷ lệ precision là {precision:.2%}. Điều này thể hiện {precision:.2%} trong số các trường hợp được dự đoán là tích cực (mắc bệnh) thì thực sự là những trường hợp mắc bệnh. "
    recall_comment = f"Tỷ lệ recall là {recall:.2%}. Điều này nói lên rằng mô hình bắt được {recall:.2%} trong số các trường hợp mắc bệnh thực sự. "
    f1_score_comment = f"F1-score đạt {f1:.2f}. Đây là một độ đo kết hợp giữa precision và recall, mang lại cái nhìn tổng quan về hiệu suất của mô hình trong việc dự đoán tình trạng sức khỏe của bệnh nhân. "
    # Bình luận về NPV
    npv_comment = f"Tỷ lệ NPV là {npv:.2%}. Điều này cho biết {npv:.2%} trong số các trường hợp được dự đoán là không mắc bệnh (tiêu cực) thì thực sự là không mắc bệnh. "
    # Bình luận về Specificity
    specificity_comment = f"Tỷ lệ specificity là {specificity:.2%}. Điều này là độ đo về khả năng mô hình loại bỏ đúng các trường hợp không mắc bệnh."

    return accuracy_comment + precision_comment + recall_comment + f1_score_comment + npv_comment + specificity_comment


if __name__ == '__main__':
    app.run(debug=True)