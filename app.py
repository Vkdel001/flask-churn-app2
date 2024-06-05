from flask import Flask, request, jsonify, send_file, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def upload_form():
    return send_from_directory('.', 'upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    data = pd.read_excel(file)
    X = data.drop(columns=['SUBSCRIBER_ID', 'CHURN'])
    y = data['CHURN']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test_scaled)[:, 1]
    new_threshold = 0.3
    y_pred_rf_adjusted = np.where(y_pred_proba_rf >= new_threshold, 1, 0)
    accuracy_rf_adjusted = accuracy_score(y_test, y_pred_rf_adjusted)
    classification_rep_rf_adjusted = classification_report(y_test, y_pred_rf_adjusted)
    churned_indices = np.where(y_pred_rf_adjusted == 1)[0]
    churned_subscriber_ids = X_test.iloc[churned_indices].index
    churned_subscriber_data = data.loc[churned_subscriber_ids, 'SUBSCRIBER_ID']
    output_file = 'churned_subscribers.xlsx'
    churned_subscriber_data.to_excel(output_file, index=False)
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
