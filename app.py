# # front end

# import os,sys

# import certifi
# ca=certifi.where()

# from dotenv import load_dotenv
# load_dotenv()
# mongo_db_url=os.getenv("MONGO_DB_URL")
# print(mongo_db_url)

# import pymongo
# from churn_prediction.exception.exception import TelecomChurnException
# from churn_prediction.logging.logger import logging
# from churn_prediction.pipeline.training_pipeline import TrainingPipeline

# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI,File,UploadFile,Request
# from uvicorn import run as app_run
# from fastapi.responses import Response
# from starlette.responses import RedirectResponse
# import pandas as pd

# from churn_prediction.utils.main_utils.utils import load_object
# from churn_prediction.utils.ml_utils.model.estimator import TelecomChurnModel
# from churn_prediction.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME,DATA_INGESTION_COLLECTION_NAME

# client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

# database=client[DATA_INGESTION_DATABASE_NAME]
# collection=client[DATA_INGESTION_COLLECTION_NAME]

# from fastapi.templating import Jinja2Templates
# #picks up all html files in templates folder
# templates=Jinja2Templates(directory="./templates")

# app=FastAPI()
# origins=["*"]

# #make sure that we access home page
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )


# @app.get("/",tags=["authentication"])
# async def index():
#     return RedirectResponse(url="/docs")

# @app.get("/train")
# async def train_route():
#     try:
#         train_pipeline=TrainingPipeline()
#         train_pipeline.run_pipeline()
#         return Response("Training Successfull")
#     except Exception as e:
#         raise TelecomChurnException(e,sys)

# @app.post("/predict")   
# async def predict_route(request: Request, file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)
#         preprocessor = load_object("final_model/preprocessor.pkl")
#         final_model = load_object("final_model/model.pkl")
#         telecom_model = TelecomChurnModel(preprocessor=preprocessor, model=final_model)
#         y_pred = telecom_model.predict(df)
#         df["predicted_column"] = y_pred
#         df.to_csv("prediction_output/output.csv")
#         table_html = df.to_html(classes="table table-striped")
#         return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
#     except Exception as e:
#         raise TelecomChurnException(e, sys)



# if __name__=="__main__":
#     app_run(app,host="localhost",port=8000)

# app.py
# app.py
import os
import pickle
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd

# Try to import your existing TrainingPipeline & TelecomChurnModel wrapper
HAS_TRAINING_PIPELINE = False
HAS_CUSTOM_MODEL_CLASS = False
try:
    from churn_prediction.pipeline.training_pipeline import TrainingPipeline
    HAS_TRAINING_PIPELINE = True
except Exception:
    HAS_TRAINING_PIPELINE = False

try:
    from churn_prediction.utils.ml_utils.model.estimator import TelecomChurnModel
    HAS_CUSTOM_MODEL_CLASS = True
except Exception:
    HAS_CUSTOM_MODEL_CLASS = False

# Config
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "prediction_output"
MODEL_DIR = BASE_DIR / "final_model"   # put model.pkl & preprocessor.pkl here
ALLOWED_EXTENSIONS = {"csv"}

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"  # for flash messages

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_artifacts():
    """
    Load preprocessor and model objects from MODEL_DIR.
    Expects preprocessor.pkl and model.pkl (or .joblib).
    Returns (preprocessor, model)
    """
    preprocessor = None
    model = None

    p1 = MODEL_DIR / "preprocessor.pkl"
    p2 = MODEL_DIR / "preprocessor.joblib"
    m1 = MODEL_DIR / "model.pkl"
    m2 = MODEL_DIR / "model.joblib"

    if p1.exists():
        with open(p1, "rb") as f:
            preprocessor = pickle.load(f)
    elif p2.exists():
        import joblib
        preprocessor = joblib.load(p2)

    if m1.exists():
        with open(m1, "rb") as f:
            model = pickle.load(f)
    elif m2.exists():
        import joblib
        model = joblib.load(m2)

    return preprocessor, model

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html", has_training_pipeline=HAS_TRAINING_PIPELINE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        input_path = UPLOAD_FOLDER / file.filename
        file.save(input_path)

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            flash(f"Could not read CSV: {e}")
            return redirect(url_for("index"))

        preprocessor, model = load_artifacts()
        if preprocessor is None or model is None:
            flash("Preprocessor and/or model not found. Place preprocessor.pkl and model.pkl in the final_model/ directory.")
            return redirect(url_for("index"))

        preds = None
        if HAS_CUSTOM_MODEL_CLASS:
            try:
                telecom_model = TelecomChurnModel(preprocessor=preprocessor, model=model)
                preds = telecom_model.predict(df)
            except Exception:
                preds = None

        if preds is None:
            try:
                X = preprocessor.transform(df)
                preds = model.predict(X)
            except Exception as e:
                flash(f"Automatic preprocessing/prediction failed: {e}")
                return redirect(url_for("index"))

        df_out = df.copy()
        df_out["predicted_churn"] = preds

        output_filename = f"predictions_{input_path.stem}.csv"
        output_path = OUTPUT_FOLDER / output_filename
        df_out.to_csv(output_path, index=False)

        table_html = df_out.to_html(classes="table table-striped", index=False, justify="center", border=0)
        return render_template("table.html", table_html=table_html, download_url=url_for("download_file", filename=output_filename))

    else:
        flash("Allowed file types: csv")
        return redirect(url_for("index"))

@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(file_path, as_attachment=True)

# NEW: route to trigger training
@app.route("/train", methods=["POST"])
def train():
    if not HAS_TRAINING_PIPELINE:
        flash("Training pipeline not available in the environment. Make sure churn_prediction.pipeline.training_pipeline.TrainingPipeline is importable.")
        return redirect(url_for("index"))

    try:
        flash("Training started. This may take a while — see server logs for progress.")
        # Run pipeline synchronously (will block until finished)
        tp = TrainingPipeline()
        tp.run_pipeline()
        flash("Training finished successfully. New model artifacts (model.pkl / preprocessor.pkl) should be in final_model/ or the artifact directory your pipeline saves to.")
    except Exception as e:
        # Prefer not to expose raw trace to users, but give a helpful message
        flash(f"Training failed: {e}")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # optional: check if model artifacts exist
    preproc, mdl = load_artifacts()
    if preproc is None or mdl is None:
        print("WARNING: preprocessor/model not found in final_model/ — place preprocessor.pkl and model.pkl there or ensure your training pipeline writes them.")
    app.run(host="0.0.0.0", port=8000, debug=True)
