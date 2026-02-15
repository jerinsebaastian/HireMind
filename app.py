from flask import Flask, render_template, request
import csv
import joblib
import sqlite3
from datetime import datetime
from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- DATABASE SETUP ----------------

def init_db():
    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            role TEXT DEFAULT 'user'
        )
    """)


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            job_role TEXT,
            total_gap INTEGER,
            readiness TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER,
            skill_name TEXT,
            user_level INTEGER,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id)
        )
    """)

    # Create default admin if not exists
    cursor.execute("SELECT * FROM users WHERE email = ?", ("admin@hiremind.com",))
    admin_exists = cursor.fetchone()

    if not admin_exists:
        from werkzeug.security import generate_password_hash
        admin_password = generate_password_hash("admin123")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, ("Admin", "admin@hiremind.com", admin_password, "admin"))


    conn.commit()
    conn.close()


# Load ML model
model = joblib.load("ml/readiness_model.pkl")
encoder = joblib.load("ml/label_encoder.pkl")

# ---------------- ROUTES ----------------

@app.route('/')
def select_role():
    if 'user_id' not in session:
        return redirect(url_for('login'))
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
    total_gap, gap_details, extra_features = calculate_skill_gap(...)
    features = [[
        total_gap,
        extra_features["missing"],
        extra_features["weak"],
        extra_features["moderate"],
        extra_features["strong"],
        extra_features["avg_level"],
        extra_features["high_importance_gap"]
    ]]
    readiness = predict_readiness_ml(features)
    recommendations = generate_recommendations(gap_details)
    # Save assessment to database
    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO assessments (user_id, job_role, total_gap, readiness, created_at)
        VALUES (?,?, ?, ?, ?)
    """, (session['user_id'], job_role, total_gap, readiness, created_at))

    assessment_id = cursor.lastrowid

    # Save individual skill entries
    for skill, level in user_skills.items():
        cursor.execute("""
            INSERT INTO user_skills (assessment_id, skill_name, user_level)
            VALUES (?, ?, ?)
        """, (assessment_id, skill, level))

    conn.commit()
    conn.close()


    return render_template(
        'result.html',
        job_role=job_role,
        total_gap=total_gap,
        readiness=readiness,
        gap_details=gap_details,
        recommendations=recommendations
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("hiremind.db")
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
            """, (name, email, hashed_password))
            conn.commit()
        except:
            return "Email already exists!"
        finally:
            conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("hiremind.db")
        cursor = conn.cursor()

        cursor.execute("SELECT id, password_hash, role FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['role'] = user[2]

            if user[2] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            return "Invalid email or password"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    # Get total users
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
    total_users = cursor.fetchone()[0]

    # Get total assessments
    cursor.execute("SELECT COUNT(*) FROM assessments")
    total_assessments = cursor.fetchone()[0]

    # Get all assessments
    cursor.execute("""
        SELECT users.name, assessments.job_role, assessments.total_gap,
               assessments.readiness, assessments.created_at
        FROM assessments
        JOIN users ON assessments.user_id = users.id
        ORDER BY assessments.created_at DESC
    """)
    all_assessments = cursor.fetchall()

        # Readiness distribution
    cursor.execute("""
        SELECT readiness, COUNT(*)
        FROM assessments
        GROUP BY readiness
    """)
    readiness_data = cursor.fetchall()

    # Job role distribution
    cursor.execute("""
        SELECT job_role, COUNT(*)
        FROM assessments
        GROUP BY job_role
    """)
    role_data = cursor.fetchall()

    conn.close()

    return render_template(
    'admin_dashboard.html',
    total_users=total_users,
    total_assessments=total_assessments,
    assessments=all_assessments,
    readiness_data=readiness_data,
    role_data=role_data
    )


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_role, total_gap, readiness, created_at
        FROM assessments
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (session['user_id'],))

    assessments = cursor.fetchall()
    conn.close()

    return render_template('dashboard.html', assessments=assessments)


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
    missing_count = 0
    weak_count = 0
    moderate_count = 0
    strong_count = 0
    high_importance_gap = 0
    total_gap = 0
    details = []

    for skill, weight in job_requirements.items():
        user_level = user_skills.get(skill, 0)
        if user_level == 3:
            status = "Strong"
            strong_count += 1
        elif user_level == 2:
            status = "Moderate"
            moderate_count += 1
        elif user_level == 1:
            status = "Weak"
            weak_count += 1
        else:
            status = "Missing"
            missing_count += 1
        if weight >= 4:
            high_importance_gap += weighted_gap
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

    avg_skill_level = sum(user_skills.values()) / len(user_skills) if user_skills else 0

    return total_gap, details, {
    "missing": missing_count,
    "weak": weak_count,
    "moderate": moderate_count,
    "strong": strong_count,
    "avg_level": avg_skill_level,
    "high_importance_gap": high_importance_gap
    }

def predict_readiness_ml(features):
    pred = model.predict(features)
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
    init_db()
    app.run(debug=True)
