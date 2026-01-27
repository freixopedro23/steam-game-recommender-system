from sqlalchemy import create_engine, Column, Integer, String, Date, Float, Text, Boolean, func
from sqlalchemy.orm import declarative_base
import os

DB_FOLDER = 'database'
DB_NAME = 'game_recommender_system.db'

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# Criando conexão
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL)

# Base para os modelos
Base = declarative_base()

class Game(Base):
    __tablename__ = 'game'

    # Identificador
    cd_game = Column(Integer, primary_key=True)
    nm_game = Column(String(100))

    # Dados Front-End
    dt_release = Column(Date, default=func.now())
    vl_price = Column(Float)
    ds_url_header = Column(String(100))
    ds_website = Column(String(100))
    ds_support_url = Column(String(100))

    # Filtros
    vl_required_age = Column(Integer)
    vl_metacritic_score = Column(Integer)
    vl_positive_ratio = Column(Float)
    vl_recommendations = Column(Integer)
    bl_windows = Column(Boolean)
    bl_mac = Column(Boolean)
    bl_linux = Column(Boolean)

    # NLP
    ds_developer = Column(Text)
    ds_publisher = Column(Text)
    ds_categories = Column(Text)
    ds_genres = Column(Text)
    ds_tags = Column(Text)
    ds_short_description = Column(Text)

    def __repr__(self):
        return f"<Game(name='`{self.nm_game}, id={self.cd_game}`')>"


class User(Base):
    __tablename__ = 'user'

    cd_user = Column(Integer, primary_key=True)
    ds_login = Column(String(100))
    ds_password = Column(String(100))

def init_db():
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    print("Criando tabelas...")
    init_db()
    print("Tabelas criadas.")