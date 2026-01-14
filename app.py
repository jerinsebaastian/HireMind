from flask import Flask, render_template, request
import csv
import joblib

app = Flask(__name__)

# Load ML model
model = joblib.load("ml/readiness_model.pkl")
encoder = joblib.load("ml/label_encoder.pkl")

# ---------------- ROUTES ----------------

@app.route('/')
def select_role():
    return render_template('select_role.html')

@app.route('/skills', methods=['POST'])
def skills():
    job_role = request.form['job_role']
    return render_template('skills.html', job_role=job_role)

@app.route('/analyze', methods=['POST'])
def analyze():
    job_role = request.form['job_role']

    user_skills = {
        "Python": int(request.form['Python']),
        "SQL": int(request.form['SQL']),
        "Machine Learning": int(request.form['Machine Learning']),
        "HTML": int(request.form['HTML']),
        "CSS": int(request.form['CSS'])
    }

    job_requirements = get_job_requirements(job_role)
    total_gap, gap_details = calculate_skill_gap(user_skills, job_requirements)
    readiness = predict_readiness_ml(total_gap)
    recommendations = generate_recommendations(gap_details)

    return render_template(
        'result.html',
        job_role=job_role,
        total_gap=total_gap,
        readiness=readiness,
        gap_details=gap_details,
        recommendations=recommendations
    )

# ---------------- LOGIC ----------------

def get_job_requirements(role):
    req = {}
    with open('data/job_roles.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['job_role'] == role:
                req[row['skill']] = int(row['weight'])
    return req

def calculate_skill_gap(user_skills, job_requirements):
    REQUIRED_LEVEL = 3
    total_gap = 0
    details = []

    for skill, weight in job_requirements.items():
        user_level = user_skills.get(skill, 0)
        gap = max(0, REQUIRED_LEVEL - user_level)
        weighted_gap = gap * weight
        total_gap += weighted_gap

        status = "Strong" if user_level == 3 else "Weak" if user_level > 0 else "Missing"

        details.append({
            "skill": skill,
            "user_level": user_level,
            "required_level": REQUIRED_LEVEL,
            "weight": weight,
            "status": status
        })

    return total_gap, details

def predict_readiness_ml(total_gap):
    pred = model.predict([[total_gap]])
    return encoder.inverse_transform(pred)[0]

def generate_recommendations(gap_details):
    recs = []
    for d in gap_details:
        if d['status'] != "Strong":
            priority = (3 - d['user_level']) * d['weight']
            recs.append({"skill": d['skill'], "priority": priority})
    return sorted(recs, key=lambda x: x['priority'], reverse=True)

# ---------------- RUN ----------------

if __name__ == '__main__':
    app.run(debug=True)
