"""Rotas de inferência do modelo."""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.prediction import ErrorResponse, PredictRequest, PredictResponse
from app.services import model_service
from app.services.model_service import BusinessRuleError

router = APIRouter(prefix="/v1", tags=["Predição"])

# Respostas documentadas no Swagger (/docs) além do 200
ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Dados válidos, mas incoerentes entre si (regra de negócio)."},
    422: {"description": "JSON fora do contrato: tipo errado, valor fora da faixa, campo faltando ou extra."},
    500: {"model": ErrorResponse, "description": "Erro interno inesperado."},
    503: {"model": ErrorResponse, "description": "Modelo não carregado na memória."},
}


def get_model_bundle(request: Request) -> dict:
    """Dependência: entrega o modelo que o lifespan deixou em memória.

    Se o carregamento falhou na inicialização, responde 503 em vez de quebrar.
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível no momento. Verifique o GET /health.",
        )
    return bundle


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=ERROR_RESPONSES,
    summary="Prevê o churn de um cliente",
)
def predict(payload: PredictRequest, bundle: dict = Depends(get_model_bundle)) -> PredictResponse:
    """Recebe os dados já validados pelo Pydantic e delega a inferência ao service.

    A rota não contém regra de modelo: ela só traduz HTTP <-> camada de serviço.
    """
    try:
        return model_service.predict(bundle, payload)
    except BusinessRuleError as exc:
        # Tipos e faixas corretos (passou no 422), mas combinação incoerente -> 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc