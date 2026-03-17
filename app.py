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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            skill_name TEXT,
            skill_level INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT UNIQUE,
            icon TEXT DEFAULT '💼'
        )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_id INTEGER,
            skill_name TEXT,
            weight INTEGER,
            FOREIGN KEY (role_id) REFERENCES job_roles(id)
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

# ----------------career paths----------------

career_paths = {

"Frontend Developer": [
"Full Stack Developer",
"UI Engineer",
"Web Architect"
],

"Backend Developer": [
"Full Stack Developer",
"DevOps Engineer",
"Software Architect"
],

"Web Developer": [
"Frontend Developer",
"Backend Developer",
"Full Stack Developer"
],

"Software Developer": [
"Software Architect",
"Tech Lead",
"Engineering Manager"
],

"Data Analyst": [
"Data Scientist",
"Business Intelligence Engineer",
"Data Engineer"
],

"Data Scientist": [
"AI Engineer",
"ML Engineer",
"AI Researcher"
],

"AI Engineer": [
"ML Architect",
"AI Research Scientist",
"AI Product Lead"
],

"DevOps Engineer": [
"Cloud Engineer",
"Site Reliability Engineer",
"Platform Engineer"
],

"QA Engineer": [
"Automation Engineer",
"Test Architect",
"QA Lead"
],

"Technical Support Engineer": [
"System Administrator",
"DevOps Engineer",
"Cloud Support Engineer"
],

"Business Analyst": [
"Product Manager",
"Data Analyst",
"Strategy Consultant"
]

}

# ---------------- ROUTES ----------------

@app.route('/')
def select_role():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("SELECT role_name, icon FROM job_roles")
    roles = cursor.fetchall()

    conn.close()

    return render_template('select_role.html', roles=roles)

@app.route('/skills', methods=['POST'])
def skills():

    job_role = request.form['job_role']

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
        WHERE user_id = ?
    """, (session['user_id'],))

    previous_assessments = cursor.fetchone()[0]
    conn.close()

    return render_template(
        "skills.html",
        job_role=job_role,
        previous_assessments=previous_assessments
    )

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

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*) FROM user_profile_skills WHERE user_id = ?
    """, (session['user_id'],))

    exists = cursor.fetchone()[0]

    if exists == 0:
        for skill, level in user_skills.items():
            cursor.execute("""
            INSERT INTO user_profile_skills (user_id, skill_name, skill_level)
            VALUES (?, ?, ?)
            """, (session['user_id'], skill, level))

    conn.commit()
    conn.close()

    job_requirements = get_job_requirements(job_role)
    total_gap, gap_details, extra_features = calculate_skill_gap(user_skills, job_requirements)
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
    role_suggestions = recommend_roles(user_skills)

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

    paths = generate_career_paths(job_role, gap_details)
    
    return render_template(
        'result.html',
        job_role=job_role,
        total_gap=total_gap,
        readiness=readiness,
        gap_details=gap_details,
        recommendations=recommendations,
        career_paths=paths,
        role_suggestions=role_suggestions
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return "Passwords do not match!"

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

            # Fetch user name
            conn = sqlite3.connect("hiremind.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = ?", (user[0],))
            name = cursor.fetchone()[0]
            conn.close()

            session['username'] = name

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

    # Manage Job Roles 
    cursor.execute("SELECT id, role_name, icon FROM job_roles")
    roles = cursor.fetchall()

    conn.close()

    return render_template(
    'admin_dashboard.html',
    total_users=total_users,
    total_assessments=total_assessments,
    assessments=all_assessments,
    readiness_data=readiness_data,
    role_data=role_data,
    roles=roles
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

@app.route('/use_previous_skills', methods=['POST'])
def use_previous_skills():

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT skill_name, skill_level
    FROM user_profile_skills
    WHERE user_id = ?
    """, (session['user_id'],))

    rows = cursor.fetchall()
    conn.close()

    skills = {row[0]: row[1] for row in rows}

    return skills

@app.route('/use_previous_skills_result')
def use_previous_skills_result():

    job_role = request.args.get('job_role')

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT skill_name, user_level
        FROM user_skills
        WHERE assessment_id = (
            SELECT id FROM assessments
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        )
    """, (session['user_id'],))

    rows = cursor.fetchall()
    conn.close()

    user_skills = {row[0]: row[1] for row in rows}

    job_requirements = get_job_requirements(job_role)

    total_gap, gap_details, extra_features = calculate_skill_gap(
        user_skills, job_requirements
    )

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

    paths = generate_career_paths(job_role, gap_details)

    role_suggestions = recommend_roles(user_skills)

    return render_template(
        'result.html',
        job_role=job_role,
        total_gap=total_gap,
        readiness=readiness,
        gap_details=gap_details,
        recommendations=recommendations,
        career_paths=paths,
        role_suggestions=role_suggestions
    )

@app.route('/save_skill_profile', methods=['POST'])
def save_skill_profile():

    data = request.json

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM user_profile_skills
    WHERE user_id = ?
    """, (session['user_id'],))

    for skill, level in data.items():

        cursor.execute("""
        INSERT INTO user_profile_skills
        (user_id, skill_name, skill_level)
        VALUES (?, ?, ?)
        """, (session['user_id'], skill, level))

    conn.commit()
    conn.close()

    return {"status":"success"}

@app.route('/edit_skills')
def edit_skills():

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT skill_name, skill_level
        FROM user_profile_skills
        WHERE user_id = ?
    """, (session['user_id'],))

    rows = cursor.fetchall()
    conn.close()

    skills = {row[0]: row[1] for row in rows}

    return render_template("edit_skills.html", skills=skills)

@app.route('/assessment/<int:assessment_id>')
def view_assessment(assessment_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    # Get assessment info
    cursor.execute("""
        SELECT job_role, total_gap, readiness
        FROM assessments
        WHERE id = ? AND user_id = ?
    """, (assessment_id, session['user_id']))

    assessment = cursor.fetchone()

    if not assessment:
        conn.close()
        return "Assessment not found"

    job_role, total_gap, readiness = assessment

    # Get skills
    cursor.execute("""
        SELECT skill_name, user_level
        FROM user_skills
        WHERE assessment_id = ?
    """, (assessment_id,))

    rows = cursor.fetchall()
    conn.close()

    user_skills = {row[0]: row[1] for row in rows}

    job_requirements = get_job_requirements(job_role)

    total_gap, gap_details, extra_features = calculate_skill_gap(
        user_skills,
        job_requirements
    )

    recommendations = generate_recommendations(gap_details)

    paths = generate_career_paths(job_role, gap_details)

    role_suggestions = recommend_roles(user_skills)

    return render_template(
        "result.html",
        job_role=job_role,
        total_gap=total_gap,
        readiness=readiness,
        gap_details=gap_details,
        recommendations=recommendations,
        career_paths=paths,
        role_suggestions=role_suggestions
    )

@app.route('/add_role', methods=['POST'])
def add_role():

    role = request.form['role']
    icon = request.form['icon']

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO job_roles (role_name, icon)
        VALUES (?, ?)
    """, (role, icon))

    conn.commit()
    conn.close()

    return redirect('/admin')

@app.route('/edit_role/<int:role_id>', methods=['POST'])
def edit_role(role_id):

    role = request.form['role']
    icon = request.form['icon']

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE job_roles
    SET role_name = ?, icon = ?
    WHERE id = ?
    """, (role, icon, role_id))

    conn.commit()
    conn.close()

    return redirect('/admin')

# ---------------- LOGIC ----------------

def get_job_requirements(role):

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT js.skill_name, js.weight
    FROM job_roles jr
    JOIN job_skills js ON jr.id = js.role_id
    WHERE jr.role_name = ?
    """, (role,))

    rows = cursor.fetchall()
    conn.close()

    return {row[0]: row[1] for row in rows}

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

        gap = max(0, REQUIRED_LEVEL - user_level)
        weighted_gap = gap * weight
        total_gap += weighted_gap

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

        details.append({
            "skill": skill,
            "user_level": user_level,
            "required_level": REQUIRED_LEVEL,
            "weight": weight,
            "status": status,
            "critical": True if weight >= 4 and user_level < 2 else False
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

def recommend_roles(user_skills):

    roles = []

    with open('data/job_roles.csv', newline='') as f:
        reader = csv.DictReader(f)

        job_requirements = {}

        for row in reader:
            role = row['job_role']
            skill = row['skill']
            weight = int(row['weight'])

            if role not in job_requirements:
                job_requirements[role] = {}

            job_requirements[role][skill] = weight

    for role, req in job_requirements.items():

        total_gap, _, _ = calculate_skill_gap(user_skills, req)

        roles.append({
            "role": role,
            "gap": total_gap
        })

    roles.sort(key=lambda x: x["gap"])

    return roles[:3]

def generate_career_paths(job_role, gap_details):

    paths = career_paths.get(job_role, [])

    results = []

    for role in paths:

        missing_skills = []

        for skill in gap_details:

            if skill["status"] != "Strong":
                missing_skills.append(skill["skill"])

        if missing_skills:
            results.append({
                "role": role,
                "improve": ", ".join(missing_skills[:2])
            })
        else:
            results.append({
                "role": role,
                "improve": "You already meet most requirements"
            })

    return results

def load_roles_from_csv():

    conn = sqlite3.connect("hiremind.db")
    cursor = conn.cursor()

    with open('data/job_roles.csv', newline='') as f:
        reader = csv.DictReader(f)

        role_map = {}

        for row in reader:
            role = row['job_role']
            skill = row['skill']
            weight = int(row['weight'])

            if role not in role_map:
                cursor.execute(
                    "INSERT OR IGNORE INTO job_roles (role_name) VALUES (?)",
                    (role,)
                )

                cursor.execute(
                    "SELECT id FROM job_roles WHERE role_name=?",
                    (role,)
                )
                role_id = cursor.fetchone()[0]
                role_map[role] = role_id

            role_id = role_map[role]

            cursor.execute("""
            INSERT INTO job_skills (role_id, skill_name, weight)
            VALUES (?, ?, ?)
            """, (role_id, skill, weight))

    conn.commit()
    conn.close()

# ---------------- RUN ----------------

if __name__ == '__main__':
    init_db()
    load_roles_from_csv()   # run once then remove later
    app.run(debug=True)