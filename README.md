# API de Previsão de Churn

API REST que estima a probabilidade de um cliente de telecom **cancelar o contrato (churn)**, devolvendo a probabilidade, a classe prevista e uma faixa de risco para que o time de retenção aja antes do cancelamento.

Projeto Final do Módulo 2 do curso de **Desenvolvimento de IA para Análise Preditiva** do programa **SCTEC**. Desenvolvido para consolidar a passagem de um modelo de Machine Learning do notebook para produção: treinamento reprodutível, serialização do pipeline completo e exposição como microsserviço com FastAPI, contratos validados pelo Pydantic, ciclo de vida otimizado, tratamento padronizado de erros e testes automatizados.

## 1. Problema

Identificar, a partir do perfil e dos serviços contratados, quais clientes de uma operadora de telecom têm maior chance de cancelar o contrato. Cada cliente pertence a uma de duas classes (cancela / não cancela), configurando um problema de **classificação binária**.

- **Dataset:** [Telco Customer Churn (IBM / Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 clientes e 21 colunas, em `data/telco_churn.csv`. Detalhes em [`data/README.md`](data/README.md).
- **Variável-alvo:** `Churn` (Yes/No). **26,5%** dos clientes cancelam, então as classes são desbalanceadas.
- **Por que importa:** reter um cliente costuma custar bem menos do que conquistar um novo. Um modelo que ordena a base por risco permite concentrar as ações de retenção (descontos, migração de plano, contato ativo) em quem tem mais chance de sair. Mais do que acertar a classe, o valor está em **ordenar bem o risco**, e por isso o critério de escolha do modelo foi o ROC-AUC.

Origem: reaproveita o projeto de **Análise de Churn com Regressão Logística** do curso (Opção A), aprimorado com pipeline completo de pré-processamento e comparação com Random Forest.

## 2. Como executar

```bash
git clone https://github.com/jhonatanalves/projetofinal_modulo2_sctec.git
cd projetofinal_modulo2_sctec

# Ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

python -m pip install -r requirements.txt

# Subir a API
python -m uvicorn app.main:app --reload
```

A API fica em `http://127.0.0.1:8000` e a documentação interativa (Swagger) em **http://127.0.0.1:8000/docs**. A raiz `/` não tem rota e responde `404`; use `/docs` ou `/health`.

O modelo treinado já acompanha o repositório (`models/model.joblib`), então não é preciso treinar para subir a API. Para reproduzir o treinamento e regenerar o modelo e as métricas:

```bash
python scripts/train.py
```

Para rodar os 17 testes automatizados:

```bash
python -m pytest -v
```

> Os comandos usam `python -m pip` / `python -m uvicorn` em vez de `pip` / `uvicorn` porque, em máquinas com antivírus ou políticas corporativas, os executáveis gerados em `Scripts/` podem ser bloqueados ("Acesso negado"). Chamar o módulo pelo Python evita o problema.

## 3. Estrutura do repositório

```
.
├── app/
│   ├── main.py                    # FastAPI, lifespan, routers e handler de erro 500
│   ├── core/config.py             # caminhos relativos e versão da API
│   ├── schemas/prediction.py      # contratos Pydantic (entrada, saída, health, erro)
│   ├── services/model_service.py  # carga do modelo, regras de negócio e inferência
│   └── routers/                   # health.py (GET /health) e prediction.py (POST /v1/predict)
├── data/telco_churn.csv           # base de dados
├── models/model.joblib            # pipeline treinado + metadados
├── notebooks/01_treinamento.ipynb # análise e justificativa de cada decisão
├── reports/metrics.json           # métricas de validação e teste
├── scripts/train.py               # re-treina e gera os artefatos
├── tests/test_api.py              # testes de 200, 400, 422, 500 e 503
└── requirements.txt
```

O desenvolvimento seguiu fluxo com uma branch por etapa (`feature/treinamento-modelo`, `feature/schemas-pydantic`, `feature/service-inferencia`, `feature/lifespan-main`, `feature/router-endpoints`, `feature/tratamento-erros`, `feature/testes-api`) e branches `fix/*` para correções, todas integradas por Pull Request e preservadas no grafo do repositório.

## 4. Pipeline e decisões técnicas

Cada decisão de modelagem está justificada nas células de texto do [notebook](notebooks/01_treinamento.ipynb); o script `scripts/train.py` reproduz o mesmo pipeline. O resumo abaixo cobre o essencial.

**Fase 1: Análise exploratória.** Dois achados guiaram o restante do pipeline. Primeiro, `TotalCharges` vem como texto porque os 11 clientes recém-contratados (`tenure = 0`) têm um espaço em branco no lugar do valor; convertidos para número, viram nulos, e daí a necessidade de **imputação**. Segundo, o desbalanceamento (cerca de 1 em cada 4 clientes cancela) torna a acurácia enganosa: um modelo que responde sempre "não cancela" acertaria 73%. Por isso a divisão é estratificada e a seleção usa ROC-AUC.

**Fase 2: Seleção de variáveis.** Das 21 colunas, ficaram 12 ligadas a tempo de casa, valores pagos, perfil e serviços: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Partner`, `Dependents`, `InternetService`, `OnlineSecurity`, `TechSupport`, `Contract`, `PaperlessBilling` e `PaymentMethod`. `customerID` é apenas identificador. Menos colunas também deixam o contrato da API mais simples para quem consome.

**Fase 3: Divisão e pré-processamento.** Divisão estratificada em **70/15/15** (treino 4.930, validação 1.056, teste 1.057), com `random_state` fixo. Todo o pré-processamento fica **dentro de um único `Pipeline`** do scikit-learn, junto com o modelo, o que evita *data leakage* (medianas e categorias são aprendidas só no treino) e garante que a API aplique exatamente as transformações do treino.

| Tipo | Imputação | Transformação |
| --- | --- | --- |
| Numérico | mediana (robusta a outliers) | `StandardScaler` |
| Categórico | valor mais frequente | `OneHotEncoder(handle_unknown="ignore")` |

**Fase 4: Treino e seleção.** Dois candidatos comparados na validação: **Regressão Logística** (linear, rápida e interpretável) e **Random Forest** (300 árvores, captura interações não lineares). Ambos usam `class_weight="balanced"` para compensar o desbalanceamento, priorizando o **recall**: para retenção, deixar de identificar um cliente que vai cancelar custa mais do que abordar um cliente que ficaria.

**Fase 5: Avaliação final.** O modelo escolhido é re-treinado com **treino + validação** (mais dados, mesmos hiperparâmetros) e avaliado **uma única vez** no teste. O pipeline e seus metadados (versão, limiar, colunas, data de treino) são serializados juntos em `models/model.joblib`.

**Fase 6: Microsserviço.**
- **Lifespan:** o `.joblib` é lido uma única vez na inicialização (`@asynccontextmanager`) e fica em `app.state`; nenhuma requisição acessa o disco. Se o arquivo não carregar, a API sobe em modo `degraded` e responde `503` na predição, em vez de cair.
- **Camadas:** `routers` só tratam HTTP, `services` só tratam modelo e regras de negócio, `schemas` definem os contratos; o service pode ser testado sem a API.
- **Validação em dois níveis:** o Pydantic barra o que está fora do contrato (tipo, faixa, categoria, campo extra) com `422`; o service barra dados válidos isoladamente, mas incoerentes entre si, com `400`.
- **Rotas síncronas (`def`):** a inferência do scikit-learn é CPU-bound; o FastAPI executa rotas `def` num pool de threads sem bloquear o event loop.
- **Erros padronizados:** toda resposta de erro segue o formato `{"detail": "..."}`, inclusive o `500`, tratado por um handler global que registra o erro no log e mantém a API no ar.

## 5. Tecnologias

| Biblioteca | Versão | Uso |
| --- | --- | --- |
| Python | 3.11+ | Linguagem |
| FastAPI | 0.162.0 | Framework da API |
| Uvicorn | 0.54.0 | Servidor ASGI |
| Pydantic | 2.13.5 | Validação dos contratos de entrada e saída |
| scikit-learn | 1.9.1 | Pipeline de pré-processamento e modelo |
| pandas / NumPy | 3.0.5 / 2.5.3 | Manipulação dos dados |
| joblib | 1.6.0 | Serialização do modelo |
| matplotlib | 3.11.2 | Gráficos do notebook |
| pytest / httpx | 9.1.1 / 0.28.1 | Testes automatizados da API |

## 6. Documentação da API

| Método | Rota | Descrição | Respostas |
| --- | --- | --- | --- |
| GET | `/health` | Serviço ativo e modelo em memória | 200 |
| POST | `/v1/predict` | Predição de churn para um cliente | 200, 400, 422, 500, 503 |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "churn-logistic_regression-v1",
  "api_version": "1.0.0"
}
```

`status` vira `"degraded"` e `model_loaded` vira `false` se o modelo não pôde ser carregado na inicialização.

### `POST /v1/predict`

Requisição:

```json
{
  "customer_id": "7590-VHVEG",
  "tenure": 12,
  "monthly_charges": 70.35,
  "total_charges": 845.50,
  "senior_citizen": false,
  "partner": "Yes",
  "dependents": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

Com `curl`:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"7590-VHVEG","tenure":12,"monthly_charges":70.35,"total_charges":845.5,"senior_citizen":false,"partner":"Yes","dependents":"No","internet_service":"Fiber optic","online_security":"No","tech_support":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check"}'
```

Resposta `200 OK`:

```json
{
  "customer_id": "7590-VHVEG",
  "churn": true,
  "churn_probability": 0.8345,
  "risk_level": "alto",
  "model_version": "churn-logistic_regression-v1"
}
```

`churn` é `true` quando a probabilidade é ≥ 0,50. `risk_level`: **baixo** (< 0,30), **médio** (< 0,60) ou **alto** (≥ 0,60). Veja na Seção 8 como interpretar esses valores.

### Regras de validação do contrato

| Campo | Tipo | Regra |
| --- | --- | --- |
| `customer_id` | texto (opcional) | 3 a 20 caracteres, letras, números e hífen |
| `tenure` | inteiro | 0 a 72 meses |
| `monthly_charges` | número | > 0 e ≤ 150 |
| `total_charges` | número | 0 a 10.000 |
| `senior_citizen` | booleano | `true` / `false` |
| `partner`, `dependents`, `paperless_billing` | texto | `"Yes"` ou `"No"` |
| `internet_service` | texto | `"DSL"`, `"Fiber optic"`, `"No"` |
| `online_security`, `tech_support` | texto | `"Yes"`, `"No"`, `"No internet service"` |
| `contract` | texto | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `payment_method` | texto | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

Os campos numéricos são **estritos**: `"12"` (texto) é rejeitado, assim como campos extras não previstos.

### Códigos de erro

| Status | Quando acontece | Exemplo |
| --- | --- | --- |
| 422 | JSON fora do contrato (validado pelo Pydantic) | `"tenure": "doze"`, `"tenure": -5`, `"contract": "Mensal"`, campo faltando ou extra |
| 400 | Campos válidos, mas incoerentes entre si | `total_charges` menor que `monthly_charges` com `tenure ≥ 1`; `internet_service: "No"` com `online_security: "Yes"` |
| 500 | Erro inesperado no servidor (resposta JSON, API continua no ar) | `{"detail": "Erro interno ao processar a requisição."}` |
| 503 | Modelo não carregado na memória | `{"detail": "Modelo indisponível no momento. Verifique o GET /health."}` |

Exemplo de resposta **400**:

```json
{
  "detail": "total_charges não pode ser menor que monthly_charges para clientes com tenure >= 1."
}
```

## 7. Resultados

**Modelo final: Regressão Logística**, escolhida pelo maior ROC-AUC na validação. Valores completos em [`reports/metrics.json`](reports/metrics.json).

| Conjunto | Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Validação | Regressão Logística | 0,735 | 0,500 | 0,821 | 0,622 | 0,840 |
| Validação | Random Forest | 0,750 | 0,519 | 0,789 | 0,626 | 0,836 |
| **Teste** | **Regressão Logística** | **0,748** | **0,518** | **0,776** | **0,621** | **0,843** |

Validação sobre 1.056 clientes; teste sobre 1.057 clientes (281 cancelamentos), limiar de decisão 0,50.

**Sobre a escolha do modelo:** os dois candidatos ficaram praticamente empatados. A Random Forest teve acurácia e precisão ligeiramente maiores, a Regressão Logística teve recall e ROC-AUC maiores, com diferença de apenas 0,004 no ROC-AUC. Nesse empate técnico, a Regressão Logística é a escolha defensável: treina e prediz mais rápido, gera um artefato menor e tem coeficientes interpretáveis, o que importa quando o time de negócio pergunta *por que* um cliente foi sinalizado.

**Sobre o trade-off precisão × recall:** no teste, o modelo sinalizou 421 clientes como churn e acertou 218 deles. Encontrou **78% dos 281 clientes que realmente cancelaram**, deixando escapar 63. O custo dessa cobertura é a precisão de 0,52: cerca de metade dos sinalizados não cancelaria. É uma troca consciente, consequência do `class_weight="balanced"`. Abordar um cliente fiel com uma oferta de retenção custa pouco; perder um cliente sem nem tentar retê-lo custa a receita inteira.

**Sobre a estabilidade:** o ROC-AUC de teste (0,843) ficou alinhado ao de validação (0,840), sinal de que o modelo não se ajustou demais aos dados de seleção e de que a estimativa de desempenho é confiável.

## 8. Análise crítica: o que a probabilidade realmente significa

Esta seção concentra a contribuição analítica do projeto.

**A probabilidade devolvida pela API está inflada.** O `class_weight="balanced"` faz o modelo treinar como se as duas classes tivessem o mesmo tamanho. Isso melhora o recall, mas desloca as probabilidades para cima. No conjunto de teste, a **probabilidade média prevista foi de 40,5%**, enquanto a **taxa real de churn foi de 26,6%**. Ou seja: o número em `churn_probability` não deve ser lido literalmente como "chance de cancelar".

**Na prática, as faixas de risco valem mais do que o número em si.** Medindo a taxa real de cancelamento dentro de cada faixa, no teste:

| Faixa | Probabilidade prevista | Clientes | Cancelaram de fato |
| --- | --- | --- | --- |
| baixo | < 0,30 | 449 | 4,5% |
| médio | 0,30 a 0,60 | 263 | 24,0% |
| alto | ≥ 0,60 | 345 | 57,4% |

O cliente de exemplo da Seção 6 recebeu `churn_probability: 0.8345`, mas entre os clientes da faixa "alto" a taxa real de cancelamento foi de 57%, não de 83%. Mesmo assim, a **ordenação** é excelente: um cliente da faixa alta tem cerca de 13 vezes mais chance de cancelar do que um da faixa baixa. É exatamente isso que o ROC-AUC de 0,84 mede, e é isso que o time de retenção precisa para priorizar quem abordar.

**A conclusão:** o modelo é bom para **ordenar** clientes por risco e fraco como **estimador de probabilidade**. Para o uso pretendido (decidir quem abordar primeiro), isso é suficiente. Para um uso que dependa do valor absoluto, como estimar a receita em risco somando probabilidades, seria preciso calibrar o modelo antes (ver Seção 9).

**O limiar de 0,50 é uma escolha, não um padrão.** Ele foi mantido por simplicidade, mas o limiar ideal depende do custo de uma ação de retenção comparado ao valor de um cliente retido, informação que pertence ao negócio, não ao modelo.

## 9. Melhorias para versões futuras

- **v2: calibração das probabilidades** (`CalibratedClassifierCV` com Platt ou isotônica), para que `churn_probability` possa ser lida literalmente. É a limitação mais evidente do projeto (Seção 8).
- **v2: limiar orientado a custo:** escolher o ponto de corte pela razão entre o custo de uma ação de retenção e o valor de um cliente retido, em vez do 0,50 fixo.
- **v3: explicabilidade por cliente:** devolver na resposta os fatores que mais pesaram (coeficientes da Regressão Logística ou SHAP), para orientar *qual* ação de retenção oferecer.
- **v3: operação:** containerização com Docker, deploy e monitoramento de *data drift*, já que o perfil dos clientes muda com o tempo e o modelo precisa ser re-treinado quando a distribuição se afastar da base de treino.
- **Modelagem:** testar as 7 variáveis descartadas (`gender`, `PhoneService`, `MultipleLines`, `OnlineBackup`, `DeviceProtection`, `StreamingTV` e `StreamingMovies`) e modelos de gradient boosting, avaliando se o ganho em ROC-AUC compensa a perda de interpretabilidade.

## 10. Autor

Jhonatan Alves
