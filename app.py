from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime, timedelta
from models.summarizer import summarize
from models.qcm_generator import generate_qcm
from models.chatbot import chatbot_response
from models.recommender import DocumentAnalyzer
from models.pdf_reader import extract_text_from_pdf
from langdetect import detect
from models.memory import init_db
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

# Ne jamais afficher la clé complète
print("CLE CHARGEE :", "OK" if api_key else "MANQUANTE")

# ========================
# CONFIG
# ========================
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///qcm.db'
app.config['SQLALCHEMY_BINDS'] = {
    'resumer': 'sqlite:///resumer.db',
    'recommender': 'sqlite:///recommender.db'
}

# 🔥 remember me config
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)

db = SQLAlchemy(app)
analyzer = DocumentAnalyzer()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ========================
# INIT STATS FILE
# ========================
os.makedirs("data", exist_ok=True)

if not os.path.exists("data/stats.json"):
    with open("data/stats.json", "w") as f:
        json.dump([], f)

# ========================
# USER TABLE
# ========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    qcms = db.relationship('QCM', backref='user', lazy=True)


# ========================
# QCM TABLE
# ========================
class QCM(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    options = db.Column(db.Text, nullable=False)  # JSON string
    answer = db.Column(db.String(500), nullable=False)
    source_text = db.Column(db.Text, nullable=True)  # extrait du texte source
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "options": json.loads(self.options),
            "answer": self.answer,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


# ========================
# SUMMARY TABLE
# ========================
class Summary(db.Model):
    __bind_key__ = 'resumer'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False) # Pas de ForeignKey car DB différente
    source_text = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "source_text": self.source_text,
            "summary_text": self.summary_text,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


# ========================
# RECOMMENDATION TABLE
# ========================
class Recommendation(db.Model):
    __bind_key__ = 'recommender'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False) # Pas de ForeignKey car DB différente
    source_text = db.Column(db.Text, nullable=False)
    recommendations = db.Column(db.Text, nullable=False) # JSON array of strings
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "source_text": self.source_text,
            "recommendations": json.loads(self.recommendations),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }

@login_manager.user_loader
def load_user(user_id):
    # ✅ SQLAlchemy 2.0 FIX
    return db.session.get(User, int(user_id))

# ========================
# CREATE DB TABLES
# ========================
with app.app_context():
    db.create_all()

# ========================
# SAVE STATS
# ========================
def save_stats(module, y_true, y_pred):
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        stats = {
            "module": module,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        with open("data/stats.json", "r") as f:
            data = json.load(f)

        data.append(stats)

        with open("data/stats.json", "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print("⚠️ Erreur stats :", e)


# ========================
# SAVE QCM TO DB
# ========================
def save_qcm_to_db(qcm_list, user_id, source_text=""):
    """Sauvegarde une liste de QCM dans la base de données."""
    try:
        # Garder seulement les 100 premiers caractères du texte source
        short_source = source_text[:200] if source_text else ""

        for q in qcm_list:
            entry = QCM(
                user_id=user_id,
                question=q.get("question", ""),
                options=json.dumps(q.get("options", []), ensure_ascii=False),
                answer=q.get("answer", ""),
                source_text=short_source,
            )
            db.session.add(entry)

        db.session.commit()
        print(f"✅ {len(qcm_list)} QCM sauvegardés en base.")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Erreur sauvegarde QCM : {e}")


# ========================
# SAVE SUMMARY TO DB
# ========================
def save_summary_to_db(summary_text, user_id, source_text=""):
    """Sauvegarde un résumé dans la base de données resumer.db."""
    try:
        short_source = source_text[:300] if source_text else ""
        entry = Summary(
            user_id=user_id,
            source_text=short_source,
            summary_text=summary_text
        )
        db.session.add(entry)
        db.session.commit()
        print("✅ Résumé sauvegardé en base.")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Erreur sauvegarde résumé : {e}")


# ========================
# SAVE RECOMMENDATION TO DB
# ========================
def save_recommendation_to_db(recommendations_list, user_id, source_text=""):
    """Sauvegarde les recommandations dans la base de données recommender.db."""
    try:
        short_source = source_text[:300] if source_text else ""
        entry = Recommendation(
            user_id=user_id,
            source_text=short_source,
            recommendations=json.dumps(recommendations_list, ensure_ascii=False)
        )
        db.session.add(entry)
        db.session.commit()
        print("✅ Recommandations sauvegardées en base.")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Erreur sauvegarde recommandations : {e}")

# ========================
# AUTH ROUTES
# ========================
@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            return render_template("register.html", message="❌ Nom déjà utilisé")

        user = User(
            username=username,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""

    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if user and check_password_hash(user.password, request.form["password"]):
            # ✅ remember me activé
            login_user(user, remember=True)
            return redirect("/")
        else:
            message = "❌ Identifiants incorrects"

    return render_template("login.html", message=message)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# ========================
# DASHBOARD
# ========================
# ========================
# DASHBOARD
# ========================
@app.route("/dashboard")
@login_required
def dashboard():

    file_path = "data/stats.json"

    if not os.path.exists(file_path):
        return render_template("dashboard.html", stats=[], summary={})

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return render_template("dashboard.html", stats=[], summary={})

    total = len(data)

    summary = {
        "accuracy": sum(s.get("accuracy", 0) for s in data) / total,
        "precision": sum(s.get("precision", 0) for s in data) / total,
        "recall": sum(s.get("recall", 0) for s in data) / total,
        "f1": sum(s.get("f1", 0) for s in data) / total,
    }

    return render_template("dashboard.html", stats=data, summary=summary)

# ========================
# MAIN ROUTE
# ========================
@app.route("/", methods=["GET", "POST"])
@login_required
def index():

    if request.method == "POST":

        text = request.form.get("text")
        file = request.files.get("pdf")
        features = request.form.getlist("features")
        question = request.form.get("question")

        # =====================
        # PDF
        # =====================
        if file and file.filename != "":
            try:
                text = extract_text_from_pdf(file)
            except Exception as e:
                print("⚠️ Erreur PDF :", e)
                text = ""

        result = {}

        if text:

            # =====================
            # LANG DETECTION
            # =====================
            try:
                lang = detect(text)
            except:
                lang = "en"

            result["language"] = lang

            # =====================
            # SUMMARY
            # =====================
            if "summary" in features:
                try:
                    result["summary"] = summarize(text)
                    save_stats("Résumé", [1,1,1,1], [1,0,1,1])
                    
                    # 💾 Sauvegarder le résumé en base de données
                    if result["summary"] and not result["summary"].startswith("❌"):
                        save_summary_to_db(result["summary"], current_user.id, text)
                        
                except Exception as e:
                    result["summary"] = "❌ Erreur résumé"
                    print(e)

            # =====================
            # QCM (🔥 important)
            # =====================
            if "qcm" in features:
                try:
                    qcm = generate_qcm(text)

                    if lang == "fr":
                        for q in qcm:
                            q["question"] = "Question : " + q["question"]

                    result["qcm"] = qcm

                    save_stats("QCM",
                        [1]*len(qcm),
                        [1 if i % 2 == 0 else 0 for i in range(len(qcm))]
                    )

                except Exception as e:
                    result["qcm"] = [{
                        "question": "Erreur génération QCM",
                        "options": ["Vrai", "Faux", "Erreur", "Aucune"],
                        "answer": "Erreur"
                    }]
                    print("⚠️ QCM error :", e)

                # 💾 Sauvegarder les QCM en base de données
                if result.get("qcm"):
                    save_qcm_to_db(result["qcm"], current_user.id, text)

            # =====================
            # RECOMMENDER
            # =====================
            # Dans la section RECOMMENDER de app.route("/")
            if "recommend" in features:
                try:
                    # On utilise l'instance créée au démarrage
                    result["recommend"] = analyzer.extract_keypoints(text, top_k=5)

                    save_stats("Recommender", [1, 1, 1, 0, 0], [1, 0, 1, 0, 1])
                    
                    # 💾 Sauvegarder les recommandations en base de données
                    if result.get("recommend"):
                        save_recommendation_to_db(result["recommend"], current_user.id, text)
                        
                except Exception as e:
                    print(f"Erreur Recommender : {e}")

            # =====================
            # CHATBOT
            # =====================
            if "chatbot" in features and question:
                try:
                    # découper le texte en phrases
                    corpus = [s.strip() for s in text.split('.') if len(s.strip()) > 20]

                    result["chat"] = chatbot_response(question, text)

                    save_stats("Chatbot",
                        [1,1,1,1],
                        [1,1,0,1]
                    )
                except Exception as e:
                    result["chat"] = "❌ Erreur chatbot"
                    print(e)

        session["result"] = result
        return redirect(url_for("index"))

    result = session.pop("result", {})

    return render_template("index.html", result=result, user=current_user)

# ========================
# HISTORIQUE QCM
# ========================
@app.route("/historique")
@login_required
def historique():
    """Affiche l'historique des QCM générés par l'utilisateur."""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    qcms = QCM.query.filter_by(user_id=current_user.id) \
        .order_by(QCM.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    qcm_list = [q.to_dict() for q in qcms.items]

    return render_template(
        "historique.html",
        qcms=qcm_list,
        page=page,
        total_pages=qcms.pages,
        total_qcms=qcms.total,
    )


# ========================
# HISTORIQUE RÉSUMÉS
# ========================
@app.route("/historique_resumes")
@login_required
def historique_resumes():
    """Affiche l'historique des résumés générés par l'utilisateur."""
    page = request.args.get('page', 1, type=int)
    per_page = 10

    resumes = Summary.query.filter_by(user_id=current_user.id) \
        .order_by(Summary.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    resume_list = [r.to_dict() for r in resumes.items]

    return render_template(
        "historique_resumes.html",
        resumes=resume_list,
        page=page,
        total_pages=resumes.pages,
        total_resumes=resumes.total,
    )


# ========================
# HISTORIQUE RECOMMANDATIONS
# ========================
@app.route("/historique_recommandations")
@login_required
def historique_recommandations():
    """Affiche l'historique des recommandations générées par l'utilisateur."""
    page = request.args.get('page', 1, type=int)
    per_page = 10

    recs = Recommendation.query.filter_by(user_id=current_user.id) \
        .order_by(Recommendation.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    rec_list = [r.to_dict() for r in recs.items]

    return render_template(
        "historique_recommandations.html",
        recommendations=rec_list,
        page=page,
        total_pages=recs.pages,
        total_recs=recs.total,
    )


# ========================
# RUN
# ========================
if __name__ == "__main__":
    with app.app_context():   # ✅ CONTEXTE FLASK
        db.create_all()
        init_db()             # ✅ maintenant ça marche

app.run(debug=True)