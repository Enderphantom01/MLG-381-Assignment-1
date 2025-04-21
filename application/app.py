from flask import Flask, render_template, jsonify, request
import joblib
from pathlib import Path

app = Flask(__name__)
app.static_folder = 'static'

# Load model
model_path = Path(__file__).resolve().parent / "gda_model.pkl"
model = joblib.load(model_path)


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
        gpa = round(prediction, 2)

        # Grade classification logic
        if gpa >= 3.5:
            grade = 'A'
        elif gpa >= 3.0:
            grade = 'B'
        elif gpa >= 2.5:
            grade = 'C'
        elif gpa >= 2.0:
            grade = 'D'
        else:
            grade = 'F'

        return jsonify({
            'predicted_gpa': gpa,
            'predicted_grade': grade,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)