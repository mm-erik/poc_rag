import os
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Mapped, mapped_column, declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb",
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
Base = declarative_base()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_message(session_id: str, user_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    with SessionLocal() as db:
        db.add(
            ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
            )
        )
        db.commit()
