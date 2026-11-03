"""Ponto de entrada da API.

Execução (na raiz do projeto):
    uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core import config
from app.routers import health, prediction
from app.services import model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na RAM uma única vez, na inicialização.

    As rotas apenas leem app.state.model_bundle: nenhuma requisição toca o disco.
    Se o carregamento falhar, a API sobe mesmo assim e o /health informa o problema.
    """
    try:
        app.state.model_bundle = model_service.load_model(config.MODEL_PATH)
    except Exception:
        logger.exception("Falha ao carregar o modelo; a API iniciará sem modelo.")
        app.state.model_bundle = None
    yield
    # Shutdown: libera a referência ao modelo
    app.state.model_bundle = None
    logger.info("Aplicação finalizada.")


DESCRIPTION = """
API REST que estima a probabilidade de **churn** (cancelamento) de clientes de telecom.

* `GET /health` — verifica se o serviço está ativo e se o modelo está em memória.
* `POST /v1/predict` — recebe os dados de um cliente e devolve a predição.

Modelo: pipeline scikit-learn treinado na base *Telco Customer Churn* (Kaggle/IBM).
"""

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(prediction.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Qualquer erro não previsto vira um 500 em JSON, com log completo no servidor.

    O cliente recebe uma mensagem genérica (sem stack trace) e a API continua no ar.
    """
    logger.exception("Erro não tratado em %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno ao processar a requisição."},
    )