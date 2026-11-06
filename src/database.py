"""
Task 5: Database Trail Module
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Features:
- Thread-safe SQLAlchemy ORM persistence layer.
- Stores permanent, queryable audit trail of every image classification:
  - Timestamp (UTC)
  - Filename
  - Verdict ('AI-Generated' / 'Real')
  - Calibrated Confidence (%)
  - Raw Model Score
  - Decision Threshold (tau*)
  - Latency (ms)
  - Model Architecture / Version
  - Inference Mode ('local' / 'remote')
  - Client IP
- Dynamic dialect support:
  - Local default: sqlite:///visionguard.db
  - Docker Compose / Production: postgresql://user:pass@db:5432/visionguard
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, desc
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("visionguard.database")

# Database URL resolution
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
# Fix legacy postgres:// dialect prefix if present
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure SQLAlchemy Engine
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
except Exception as e:
    logger.warning(f"Failed to create engine with {DATABASE_URL}: {e}. Falling back to sqlite:///visionguard.db")
    DATABASE_URL = "sqlite:///visionguard.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionFactory)

Base = declarative_base()
Base.query = db_session.query_property()


class PredictionRecord(Base):
    """
    SQLAlchemy ORM Model representing an immutable audit log entry
    for an image classification request.
    """
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    prediction = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    raw_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False, default=0.23)
    model_name = Column(String(100), nullable=False)
    inference_mode = Column(String(20), nullable=False, default="local")
    inference_time_ms = Column(Float, nullable=False)
    client_ip = Column(String(50), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes record to dictionary for JSON APIs and Jinja2 templates."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2),
            "raw_score": round(self.raw_score, 6),
            "threshold": round(self.threshold, 4),
            "model_name": self.model_name,
            "inference_mode": self.inference_mode,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "client_ip": self.client_ip or "127.0.0.1"
        }

    def __repr__(self) -> str:
        return f"<PredictionRecord #{self.id} {self.prediction} ({self.confidence:.1f}%) '{self.filename}'>"


def init_db(target_engine=None):
    """Initializes and creates all database tables if they do not exist."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    logger.info(f"[+] Database initialized with URL: {eng.url}")


def log_prediction_to_db(record_data: Dict[str, Any], session=None) -> Optional[int]:
    """
    Inserts a newly completed classification record into the database.
    Returns the integer primary key ID of the created record.
    """
    sess = session or db_session()
    try:
        new_record = PredictionRecord(
            filename=record_data.get("filename", "unknown.jpg"),
            prediction=record_data.get("prediction", "Unknown"),
            confidence=float(record_data.get("confidence", 0.0)),
            raw_score=float(record_data.get("raw_score", 0.0)),
            threshold=float(record_data.get("threshold", 0.23)),
            model_name=record_data.get("model_name", "VisionGuard_MobileNetV2"),
            inference_mode=record_data.get("inference_mode", "local"),
            inference_time_ms=float(record_data.get("inference_time_ms", 0.0)),
            client_ip=record_data.get("client_ip", "127.0.0.1")
        )
        sess.add(new_record)
        sess.commit()
        record_id = new_record.id
        logger.info(f"[+] Logged prediction #{record_id} to database: {new_record.prediction} ({new_record.filename})")
        return record_id
    except Exception as e:
        sess.rollback()
        logger.error(f"[-] Database insertion failed: {e}")
        return None
    finally:
        if session is None:
            sess.close()


def get_prediction_history(limit: int = 100, session=None) -> List[Dict[str, Any]]:
    """
    Retrieves the latest prediction records ordered by timestamp descending.
    """
    sess = session or db_session()
    try:
        records = sess.query(PredictionRecord).order_by(desc(PredictionRecord.timestamp)).limit(limit).all()
        return [r.to_dict() for r in records]
    except Exception as e:
        logger.error(f"[-] Database query failed: {e}")
        return []
    finally:
        if session is None:
            sess.close()


def get_record_count(session=None) -> int:
    """Returns the total number of prediction records stored in the database."""
    sess = session or db_session()
    try:
        return sess.query(PredictionRecord).count()
    except Exception as e:
        logger.error(f"[-] Record count query failed: {e}")
        return 0
    finally:
        if session is None:
            sess.close()


def get_schema_ddl() -> str:
    """Exports the exact SQL DDL schema for the prediction_records table."""
    from sqlalchemy.schema import CreateTable
    return str(CreateTable(PredictionRecord.__table__).compile(engine))

