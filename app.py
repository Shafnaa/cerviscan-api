"""
Documentation for app.py
=========================

This Python application is built using Flask and performs image processing and user management tasks.
It allows registered users to upload images for processing and view the results. The processed images
and features are stored in a database, along with a prediction for each uploaded image.

Key Features:
- User registration and authentication.
- Image upload and unique filename generation using UUID.
- Image processing pipeline including RGB to grayscale conversion, segmentation, and feature extraction.
- Prediction using a pre-trained model.
- History management to view and delete previous uploads.

Modules and Libraries Used:
- Flask: Web framework.
- Flask_SQLAlchemy: ORM for database operations.
- Flask_Login: User session management.
- Flask_Migrate: Database migration tool.
- OpenCV (cv2): Image processing.
- Matplotlib: Image saving for processed results.
- Pickle: Loading pre-trained models.
- Werkzeuge: Secure file handling.
- UUID: Unique filename generation.
- Datetime: Timestamp handling.
"""

import os
import uuid
import pytz
import base64
import io

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from flask import Flask
from flask import jsonify
from flask import request

from flask import render_template
from flask import redirect
from flask import flash
from flask import url_for

from flask_cors import CORS
from flask_cors import cross_origin

from flask_login import UserMixin

from flask_migrate import Migrate

from flask_sqlalchemy import SQLAlchemy

from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
from flask_jwt_extended import jwt_required
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import set_access_cookies
from flask_jwt_extended import get_jwt
from flask_jwt_extended import unset_jwt_cookies
from flask_jwt_extended import get_csrf_token

from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash

import cv2
import matplotlib.pyplot as plt
import pickle
from PIL import Image

from model.rgb_to_gray import rgb_to_gray_converter
from model.multiotsu_segmentation import multiotsu_masking
from model.bitwise_operation import get_segmented_image
from model.single_image_extractor import SingleImageFeatureExtractor

app = Flask(__name__)
CORS(
    app,
    # supports_credentials=True,
)

# Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sqlite.db"
app.config["SECRET_KEY"] = "your_strong_secret_key"
app.config["JWT_SECRET_KEY"] = "your_strong_secret_key"
app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config["CORS_HEADERS"] = "Content-Type"
# Folder Configurationapa itu cagr
app.config["UPLOAD_FOLDER"] = "./static/process/upload"
app.config["GRAY_FOLDER"] = "./static/process/gray"
app.config["MASK_FOLDER"] = "./static/process/mask"
app.config["SEGMENTED_FOLDER"] = "./static/process/segmented"
app.config["FEATURE_FOLDER"] = "./static/process/feature"

# Ensure directories exist for uploaded and processed files
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["GRAY_FOLDER"], exist_ok=True)
os.makedirs(app.config["MASK_FOLDER"], exist_ok=True)
os.makedirs(app.config["SEGMENTED_FOLDER"], exist_ok=True)
os.makedirs(app.config["FEATURE_FOLDER"], exist_ok=True)

# Database Initialization
db = SQLAlchemy(app)

# JWT Initialization
jwt = JWTManager(app)

Migrate(app, db)


class Users(db.Model, UserMixin):
    id = db.Column(db.String(), primary_key=True, default=str(uuid.uuid4()))
    username = db.Column(db.String(), unique=True, nullable=False)
    password = db.Column(db.String(), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"


class Records(db.Model):
    id = db.Column(db.String(), primary_key=True, default=str(uuid.uuid4()))
    user_id = db.Column(db.String(), db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(tz=pytz.timezone("UTC")))

    def __repr__(self):
        return f"<Record {self.name}>"


# Initialize database within the application context
with app.app_context():
    db.create_all()


@app.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            set_access_cookies(response, access_token)
        return response
    except (RuntimeError, KeyError):
        return response


# Authentication routes
@app.route("/api/auth/logout", methods=["POST"])
@cross_origin()
@jwt_required()
def api_logout():
    response = jsonify(message="Logout successful")
    unset_jwt_cookies(response)
    return response


@app.route("/api/auth/login", methods=["POST"])
@cross_origin()
def api_login():
    try:
        username = request.form.get("username")
        password = request.form.get("password")

        user = Users.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password, password):
            return jsonify(message="Invalid username or password"), 401

        access_token = create_access_token(identity=user.id)

        refresh_token = get_csrf_token(access_token)

        response = jsonify(
            message="Login successful",
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
        )

        response.status_code = 201

        set_access_cookies(response, access_token)

        return response

    except AttributeError:
        return jsonify(message="Provide a username and password in form data"), 400


@app.route("/api/auth/register", methods=["POST"])
@cross_origin()
def api_register():
    try:
        username = request.form.get("username")
        password = request.form.get("password")
        password_confirm = request.form.get("password_confirm")

        if not username or not password or not password_confirm:
            return jsonify(message="Please fill all fields"), 400

        if password != password_confirm:
            return jsonify(message="Passwords do not match"), 400

        if Users.query.filter_by(username=username).first():
            return jsonify(message="Username already exists"), 409

        hashed_password = generate_password_hash(password)
        new_user = Users(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        access_token = create_access_token(identity=new_user.id)

        refresh_token = get_csrf_token(access_token)

        response = jsonify(
            message="Registration successful",
            data={"access_token": access_token, "refresh_token": refresh_token},
        )

        response.status_code = 201

        set_access_cookies(response, access_token)

        return response

    except AttributeError:
        return (
            jsonify(
                message="Provide a username, password, and password_confirm in form data"
            ),
            400,
        )


# Record routes
@app.route("/api/record", methods=["GET"])
@jwt_required()
def api_list_records():
    user_id = get_jwt_identity()

    return (
        jsonify(
            data=[
                {
                    "id": record.id,
                    "name": record.name,
                    "dob": record.dob,
                    "prediction": record.prediction,
                    "created_at": record.created_at,
                }
                for record in Records.query.filter_by(user_id=user_id).all()
            ],
            message="Records retrieved successfully",
        ),
        200,
    )


@app.route("/api/record/create", methods=["POST"])
@jwt_required()
def api_create_record():
    try:
        name = request.form.get("name")
        dob = request.form.get("dob")

        user_id = get_jwt_identity()

        record_id = str(uuid.uuid4())

        while Records.query.filter_by(id=record_id).first():
            record_id = str(uuid.uuid4())

        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
            
            # Resize image while maintaining aspect ratio
            base_width = 512
            w_percent = (base_width / float(image.size[0]))
            h_size = int((float(image.size[1]) * float(w_percent)))
            image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)

            # Convert to RGB if it has an alpha channel to save as JPG
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            filename = record_id + os.path.splitext(file.filename)[1]
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(original_path)
        elif "image" in request.form:
            file = base64.b64decode(request.form.get("image"))
            image = Image.open(io.BytesIO(file))
            filename = record_id + ".jpg"
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(original_path)

        if original_path:
            gray_image = rgb_to_gray_converter(original_path)
            gray_path = os.path.join(app.config["GRAY_FOLDER"], filename)
            cv2.imwrite(gray_path, gray_image)

            mask_image = multiotsu_masking(gray_path)
            mask_path = os.path.join(app.config["MASK_FOLDER"], filename)
            plt.imsave(mask_path, mask_image, cmap="gray")

            original_image = cv2.imread(original_path)
            segmented_image = get_segmented_image(original_image, mask_path)
            segmented_path = os.path.join(app.config["SEGMENTED_FOLDER"], filename)
            cv2.imwrite(segmented_path, segmented_image)

            extractor = SingleImageFeatureExtractor(
                color_moment_features={
                    "YUV": {
                        "y": ["mean", "std", "skew"],
                        "u": ["mean", "std", "skew"],
                        "v": ["mean", "std"],
                    }
                },
                texture_features={
                    "GLRLM": {
                        "deg0": [
                            "SRE",
                            "LRE",
                            "RLN",
                            "SRLGLE",
                            "LRHGLE",
                        ],
                        "deg45": [
                            "SRE",
                            "LRE",
                            "GLN",
                            "HGL",
                        ],
                        "deg90": [
                            "SRE",
                            "LRE",
                            "LGLRE",
                        ],
                        "deg135": [
                            "LRE",
                        ],
                    },
                    "TAMURA": ["coarseness", "contrast", "directionality"],
                    "LBP": ["mean", "std"],
                },
            )

            image_features = extractor.extract_features(segmented_path)

            model = pickle.load(open("./model/xgb_best", "rb"))
            prediction = model.predict(image_features)

            entry = Records(
                id=record_id,
                user_id=user_id,
                name=name,
                dob=dob,
                prediction=bool(prediction[0]),
            )

            db.session.add(entry)
            db.session.commit()

            return (
                jsonify(
                    message="Record created successfully",
                    data={"id": record_id, "prediction": bool(prediction[0])},
                ),
                201,
            )

        return jsonify(message="No image uploaded"), 400

    except AttributeError:
        return jsonify(message="Provide a name, dob, and image in form data"), 400


@app.route("/api/record/<record_id>", methods=["DELETE"])
@jwt_required()
def api_delete_record(record_id):
    user_id = get_jwt_identity()

    record = Records.query.filter_by(id=record_id, user_id=user_id).first()

    if record:
        try:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], record_id + ".jpg"))
            os.remove(os.path.join(app.config["GRAY_FOLDER"], record_id + ".jpg"))
            os.remove(os.path.join(app.config["MASK_FOLDER"], record_id + ".jpg"))
            os.remove(os.path.join(app.config["SEGMENTED_FOLDER"], record_id + ".jpg"))
        except FileNotFoundError:
            pass

        db.session.delete(record)
        db.session.commit()
        return jsonify(message="Record deleted successfully"), 201

    return jsonify(message="Record not found"), 404


@app.route("/api/record/<record_id>", methods=["GET"])
@jwt_required()
def api_get_record(record_id):
    user_id = get_jwt_identity()

    record = Records.query.filter_by(id=record_id, user_id=user_id).first()

    if record:
        return (
            jsonify(
                data={
                    "id": record.id,
                    "name": record.name,
                    "dob": record.dob,
                    "prediction": record.prediction,
                    "created_at": record.created_at,
                }
            ),
            200,
        )

    return jsonify(message="Record not found"), 404


# # User login route
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


# # User registration route
@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


# # User logout route
@app.route("/logout", methods=["GET"])
@jwt_required()
def logout():
    response = redirect(url_for("login_page"))

    unset_jwt_cookies(response)

    return response


# # Main application route for image processing
@app.route("/", methods=["GET"])
@jwt_required(optional=True)
def index():
    user_id = get_jwt_identity()

    if not user_id:
        return redirect(url_for("login_page"))

    return render_template("index.html")


# # Route to view user history
@app.route("/history", methods=["GET"])
@jwt_required()
def history_page():
    user_id = get_jwt_identity()

    user_history = Records.query.filter_by(user_id=user_id).all()
    return render_template("history.html", history=user_history)


# # Route to view detailed history entry
@app.route("/history/<id>", methods=["GET"])
@jwt_required()
def history_detail(id):
    user_id = get_jwt_identity()

    entry = Records.query.get_or_404(id)
    if entry.user_id != user_id:
        flash("You are not authorized to view this entry", "danger")
        return redirect(url_for("history_page"))
    return render_template("detail.html", entry=entry)


# # Route to delete a history entry
@app.route("/delete/<id>", methods=["POST"])
@jwt_required()
def delete_history(id):
    user_id = get_jwt_identity()

    entry = Records.query.get_or_404(id)
    if entry.user_id != user_id:
        flash("You are not authorized to delete this entry", "danger")
        return redirect(url_for("history_page"))
    db.session.delete(entry)
    db.session.commit()
    flash("History deleted successfully.", "success")
    return redirect(url_for("history_page"))


@app.route("/usage")
@jwt_required()
def usage():
    return render_template("usage.html")


if __name__ == "__main__":
    with app.app_context():
        app.run(debug=True, host="0.0.0.0", port=8000)
