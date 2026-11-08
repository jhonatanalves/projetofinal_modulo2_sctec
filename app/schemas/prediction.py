"""Contratos de dados da API (entrada e saída), validados pelo Pydantic.

Qualquer violação destas regras é barrada pelo FastAPI com HTTP 422,
antes de chegar à camada de serviço ou ao modelo.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Categorias aceitas: exatamente as que existem na base de treino
SimNao = Literal["Yes", "No"]
ServicoAdicional = Literal["Yes", "No", "No internet service"]
TipoInternet = Literal["DSL", "Fiber optic", "No"]
TipoContrato = Literal["Month-to-month", "One year", "Two year"]
FormaPagamento = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

EXEMPLO_ENTRADA = {
    "customer_id": "7590-VHVEG",
    "tenure": 12,
    "monthly_charges": 70.35,
    "total_charges": 845.50,
    "senior_citizen": False,
    "partner": "Yes",
    "dependents": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


class PredictRequest(BaseModel):
    """Dados de um cliente para estimar a probabilidade de churn."""

    model_config = ConfigDict(
        extra="forbid",  # campo desconhecido no JSON -> 422 (evita erros de digitação silenciosos)
        json_schema_extra={"examples": [EXEMPLO_ENTRADA]},
    )

    customer_id: str | None = Field(
        default=None, min_length=3, max_length=20, pattern=r"^[A-Za-z0-9-]+$",
        description="Identificador do cliente (opcional, apenas devolvido na resposta).",
    )
    tenure: int = Field(
        ..., ge=0, le=72, strict=True,
        description="Meses como cliente (0 a 72).",
    )
    monthly_charges: float = Field(
        ..., gt=0, le=150, strict=True,
        description="Valor da mensalidade em USD (maior que 0 e até 150).",
    )
    total_charges: float = Field(
        ..., ge=0, le=10_000, strict=True,
        description="Total já pago pelo cliente em USD (0 a 10.000).",
    )
    senior_citizen: bool = Field(
        ..., strict=True, description="Cliente com 65 anos ou mais (true/false).",
    )
    partner: SimNao = Field(..., description="Possui cônjuge/parceiro.")
    dependents: SimNao = Field(..., description="Possui dependentes.")
    internet_service: TipoInternet = Field(..., description="Tipo de internet contratada.")
    online_security: ServicoAdicional = Field(..., description="Serviço de segurança online.")
    tech_support: ServicoAdicional = Field(..., description="Serviço de suporte técnico.")
    contract: TipoContrato = Field(..., description="Tipo de contrato.")
    paperless_billing: SimNao = Field(..., description="Fatura digital.")
    payment_method: FormaPagamento = Field(..., description="Forma de pagamento.")

class PredictResponse(BaseModel):
    """Resultado da predição devolvido pela API."""

    customer_id: str | None = Field(default=None, description="Mesmo identificador enviado na entrada.")
    churn: bool = Field(..., description="true se o modelo prevê cancelamento.")
    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probabilidade estimada de churn (0 a 1).",
    )
    risk_level: Literal["baixo", "médio", "alto"] = Field(
        ..., description="Faixa de risco: baixo (< 0,30), médio (< 0,60) ou alto.",
    )
    model_version: str = Field(..., min_length=1, description="Versão do modelo que gerou a predição.")


class HealthResponse(BaseModel):
    """Estado operacional do serviço."""

    status: Literal["ok", "degraded"] = Field(
        ..., description="ok = pronto para predizer; degraded = API no ar, mas sem modelo.",
    )
    model_loaded: bool = Field(..., description="Indica se o modelo está carregado na memória RAM.")
    model_version: str | None = Field(default=None, description="Versão do modelo em memória.")
    api_version: str = Field(..., description="Versão da API.")


class ErrorResponse(BaseModel):
    """Formato padrão das respostas de erro 400, 500 e 503."""

    detail: str = Field(..., description="Mensagem explicando o erro.")
