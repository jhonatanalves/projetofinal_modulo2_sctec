"""Testes da API com o TestClient do FastAPI.

Rodar na raiz do projeto:  pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.prediction import EXEMPLO_ENTRADA


@pytest.fixture(scope="module")
def client():
    # O "with" dispara o lifespan, então o modelo é carregado como em produção.
    # raise_server_exceptions=False deixa o handler de erro 500 responder normalmente.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def payload(**alteracoes) -> dict:
    """Cópia do exemplo válido com alguns campos alterados."""
    return {**EXEMPLO_ENTRADA, **alteracoes}


def test_health_retorna_200_com_modelo_carregado(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_loaded"] is True


def test_predict_valido_retorna_200(client):
    resposta = client.post("/v1/predict", json=payload())
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"customer_id", "churn", "churn_probability", "risk_level", "model_version"}
    assert 0 <= corpo["churn_probability"] <= 1
    assert corpo["risk_level"] in {"baixo", "médio", "alto"}
    assert corpo["customer_id"] == EXEMPLO_ENTRADA["customer_id"]

    # ---------- 422: barrados pelo Pydantic antes de chegar ao modelo ----------

@pytest.mark.parametrize(
    "alteracoes",
    [
        {"tenure": "doze"},                 # texto em campo numérico
        {"tenure": "12"},                   # número como texto (strict=True)
        {"tenure": -5},                     # abaixo do mínimo (ge=0)
        {"tenure": 100},                    # acima do máximo (le=72)
        {"monthly_charges": 0},             # gt=0
        {"contract": "Mensal"},             # categoria fora do Literal
        {"senior_citizen": "sim"},          # bool estrito
        {"customer_id": "ab"},              # min_length=3
        {"customer_id": "id com espaço"},   # pattern
    ],
)
def test_dados_invalidos_retornam_422(client, alteracoes):
    resposta = client.post("/v1/predict", json=payload(**alteracoes))
    assert resposta.status_code == 422


def test_campo_extra_retorna_422(client):
    resposta = client.post("/v1/predict", json=payload(campo_inexistente=1))
    assert resposta.status_code == 422


def test_campo_obrigatorio_faltando_retorna_422(client):
    dados = payload()
    del dados["tenure"]
    resposta = client.post("/v1/predict", json=dados)
    assert resposta.status_code == 422


# ---------- 400: dados válidos, mas incoerentes (regra de negócio) ----------

def test_total_menor_que_mensalidade_retorna_400(client):
    resposta = client.post("/v1/predict", json=payload(tenure=10, monthly_charges=80.0, total_charges=50.0))
    assert resposta.status_code == 400
    assert "total_charges" in resposta.json()["detail"]


def test_servico_sem_internet_incoerente_retorna_400(client):
    resposta = client.post("/v1/predict", json=payload(internet_service="No", online_security="Yes"))
    assert resposta.status_code == 400


# ---------- 500 e 503: falhas do servidor tratadas, sem derrubar a API ----------

def test_erro_inesperado_retorna_500_em_json(client, monkeypatch):
    def falha(*args, **kwargs):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("app.services.model_service.predict", falha)
    resposta = client.post("/v1/predict", json=payload())
    assert resposta.status_code == 500
    assert resposta.json() == {"detail": "Erro interno ao processar a requisição."}


def test_modelo_indisponivel_retorna_503(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "model_bundle", None)
    assert client.post("/v1/predict", json=payload()).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False