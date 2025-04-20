from flask import Flask, render_template, jsonify, request
import joblib
from pathlib import Path

app = Flask(__name__)
app.static_folder = 'static'

# Load model
model = joblib.load(Path("gda_model.pkl"))

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['StudyTimeWeekly']),
            float(data['Absences']),
            float(data['Tutoring']),
            float(data['ParentalSupport']),
            float(data['EngagementIndex']),
            float(data['AttendanceRate'])
        ]
        prediction = model.predict([features])[0]
        return jsonify({
            'predicted_gpa': round(prediction, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)