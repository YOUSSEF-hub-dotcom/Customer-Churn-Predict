import jwt
import mlflow.pyfunc
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings
import logging

logging.basicConfig(
    filename="app_activity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger("api")


# DATABASE
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CustomerPrediction(Base):
    __tablename__ = "customer_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    Contract = Column(String)
    tenure = Column(Integer)
    InternetService = Column(String)
    MonthlyCharges = Column(Float)
    PaymentMethod = Column(String)
    TechSupport_OnlineSecurity = Column(String)
    TotalCharges = Column(Float)

    churn_prediction = Column(Integer)
    churn_probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    info = Column(Text)
    prediction = Column(Integer) # 0 or 1
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# Schema

class ChurnInput(BaseModel):
    Contract: str
    tenure: int = Field(..., ge=0)
    InternetService: str
    MonthlyCharges: float = Field(..., gt=0)
    PaymentMethod: str
    TechSupport_OnlineSecurity: str
    TotalCharges: float = Field(..., ge=0)

class ChurnOutput(BaseModel):
    churn_prediction: int
    churn_probability: float
    threshold_used: float

class ErrorResponse(BaseModel):
    detail: str


try:
    mlflow_model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    print(f" MLflow Model Loaded from: {settings.MODEL_URI}")
except Exception as e:
    mlflow_model = None
    print(f" Model Load Error: {e}")

# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Rate Limiting
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"


# Intialize APP
limiter = Limiter(key_func=get_smart_identifier)
app = FastAPI(title="Customer Churn Prediction API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,                 
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response



# Feature Adapter

def build_full_feature_vector(user_data: dict) -> pd.DataFrame:
    tech_yes = "Yes" in user_data["TechSupport_OnlineSecurity"]

    full_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": user_data["tenure"],
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": user_data["InternetService"],
        "OnlineSecurity": "Yes" if tech_yes else "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes" if tech_yes else "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": user_data["Contract"],
        "PaperlessBilling": 1,
        "PaymentMethod": user_data["PaymentMethod"],
        "MonthlyCharges": user_data["MonthlyCharges"],
        "TotalCharges": user_data["TotalCharges"],
        "NumServices": 1,
        "TechSupport_OnlineSecurity": user_data["TechSupport_OnlineSecurity"]
    }

    return pd.DataFrame([full_data])


import time  # تأكد إنك عامل import للمكتبة فوق


@app.post(
    "/predict",
    response_model=ChurnOutput,
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
@limiter.limit("10/minute") # المكتبة دي بتجبرك تحط request تحتها
def predict_churn(
        input_data: ChurnInput,
        request: Request,  # <--- لازم دي تكون موجودة هنا عشان الـ Limiter
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
):
    start_time = time.time()

    if mlflow_model is None:
        logger.error(" Model requested but not loaded")
        raise HTTPException(status_code=503, detail="ML Model is not loaded or unavailable")

    try:
        # تحويل البيانات
        df = build_full_feature_vector(input_data.dict())

        # التوقع باستخدام MLflow
        try:
            result = mlflow_model.predict(df).iloc[0]
        except Exception as e:
            logger.error(f" Inference Error: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Model Inference Error: {str(e)}")

        churn_pred = int(result["churn_prediction"])
        churn_prob = float(result["churn_probability"])

        try:
            record = CustomerPrediction(
                Contract=input_data.Contract,
                tenure=input_data.tenure,
                InternetService=input_data.InternetService,
                MonthlyCharges=input_data.MonthlyCharges,
                PaymentMethod=input_data.PaymentMethod,
                TechSupport_OnlineSecurity=input_data.TechSupport_OnlineSecurity,
                TotalCharges=input_data.TotalCharges,
                churn_prediction=churn_pred,
                churn_probability=churn_prob
            )
            db.add(record)
            db.commit()
        except Exception as db_err:
            db.rollback()
            logger.warning(f"Database logging failed: {db_err}")

        execution_time = round(time.time() - start_time, 4)

        # إضافة مهمة خلفية للسجل العام
        log_summary = f"Contract: {input_data.Contract} | MonthlyCharges: {input_data.MonthlyCharges}"
        background_tasks.add_task(db_log_prediction, log_summary, churn_pred, churn_prob)

        logger.info(f" Prediction success | Prob: {churn_prob:.2f} | Time: {execution_time}s")

        return {
            "churn_prediction": churn_pred,
            "churn_probability": churn_prob,
            "threshold_used": 0.5,
            "latency_seconds": execution_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f" Unexpected Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def db_log_prediction(input_info: str, prediction: int, probability: float):
    new_db = SessionLocal()
    try:
        log_entry = PredictionLog(
            info=input_info,
            prediction=prediction,
            probability=probability
        )
        new_db.add(log_entry)
        new_db.commit()
        logger.info(f"Background Log Saved: Prediction={prediction} | Info={input_info}")

    except Exception as e:
        logger.error(f" Background Log Error: {str(e)}", exc_info=True)

    finally:
        new_db.close()

# CRUD
@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(CustomerPrediction).order_by(CustomerPrediction.created_at.desc()).limit(10).all()


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(CustomerPrediction).filter(
        CustomerPrediction.id == prediction_id
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred


@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(CustomerPrediction).filter(
        CustomerPrediction.id == prediction_id
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(pred)
    db.commit()
    return {"message": "Deleted successfully"}


@app.get("/")
def health():
    return {"status": "API is running "}
