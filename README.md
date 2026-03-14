# Credit Scoring Boilerplate (Educational)

Projeto educacional de credit scoring com Machine Learning, inspirado em fluxos de pontuação de risco usados em bureaus de crédito como Serasa Experian PowerCurve.

---

## Como o modelo funciona

### Problema

O modelo resolve um problema de **classificação binária**: dado o perfil financeiro de um cliente, estimar a probabilidade de ele entrar em **default** (inadimplência).

- Saída `1`: cliente tende a ser inadimplente.
- Saída `0`: cliente tende a honrar seus compromissos.

### Algoritmo: Regressão Logística

O algoritmo escolhido é a **Regressão Logística** (`sklearn.linear_model.LogisticRegression`), que é amplamente utilizado em aplicações de crédito por ser:

- Interpretável — permite inspecionar os pesos de cada feature.
- Calibrado — produz probabilidades confiáveis diretamente.
- Eficiente — treina e executa rapidamente mesmo com poucos dados.

### Pipeline de transformação

O modelo é um `sklearn.pipeline.Pipeline` com duas etapas:

```
StandardScaler  →  LogisticRegression
```

1. **`StandardScaler`**: normaliza cada feature para média 0 e desvio padrão 1. Isso é necessário porque as features têm escalas muito diferentes (`income` está na casa dos milhares, `payment_delays` na casa das dezenas).
2. **`LogisticRegression`**: aprende os pesos de cada feature e aplica a função sigmoide para produzir a probabilidade de default.

### Features de entrada

| Feature           | Tipo    | Descrição                          |
| ----------------- | ------- | ---------------------------------- |
| `age`             | inteiro | Idade do cliente em anos           |
| `income`          | decimal | Renda mensal                       |
| `number_of_loans` | inteiro | Quantidade de empréstimos ativos   |
| `payment_delays`  | inteiro | Quantidade de pagamentos em atraso |

### Variável alvo

| Coluna    | Valor | Significado                    |
| --------- | ----- | ------------------------------ |
| `default` | `1`   | Cliente ficou inadimplente     |
| `default` | `0`   | Cliente não ficou inadimplente |

### Saída do modelo

O endpoint retorna a **probabilidade de inadimplência**:

- Valor próximo de `0.0` → baixo risco.
- Valor próximo de `1.0` → alto risco.

```json
{ "probability_default": 0.05 }
```

### Serialização

Após o treino o pipeline completo (scaler + modelo) é salvo em `models/credit_model.pkl` via **joblib**. A API carrega esse artefato na inicialização e o mantém em memória para servir predições sem latência de I/O.

---

## Tech Stack

- Python 3.11
- FastAPI
- scikit-learn (Logistic Regression)
- pandas and numpy
- joblib
- Gemini API (camada conversacional opcional)
- Docker

## Project Structure

```text
project-root/
  frontend/
    index.html
    styles.css
    app.js
  data/
    credit_dataset.csv
  training/
    __init__.py
    config.py
    data_loader.py
    pipeline.py
    trainer.py
    train_model.py
  api/
    __init__.py
    core/
      __init__.py
      config.py
      model_store.py
    routers/
      __init__.py
      scoring.py
    services/
      __init__.py
      llm_service.py
      scoring_service.py
    schemas/
      __init__.py
      chat.py
      scoring.py
    main.py
  models/
    credit_model.pkl
  .env.example
  requirements.txt
  Dockerfile
  docker-compose.yml
  README.md
```

### Estrutura da API

- `api/main.py`: cria a aplicacao FastAPI e registra os routers.
- `api/routers/scoring.py`: endpoints `/score` e `/score/chat`.
- `api/services/scoring_service.py`: logica de predicao e explicacao de risco.
- `api/services/llm_service.py`: integracao com Gemini e extracao estruturada.
- `api/services/chat_memory_service.py`: memoria simples em processo por conversa.
- `api/schemas/*`: contratos de request/response com Pydantic.
- `api/core/*`: configuracoes compartilhadas e carregamento do modelo em memoria.

### Estrutura de treinamento

- `training/train_model.py`: ponto de entrada para treino.
- `training/trainer.py`: orquestra split, treino, avaliacao e persistencia.
- `training/pipeline.py`: define pipeline `StandardScaler + LogisticRegression`.
- `training/data_loader.py`: leitura/validacao do CSV.
- `training/config.py`: caminhos e hiperparametros centralizados.

### Estrutura do frontend

- `frontend/index.html`: tela web minima de chat.
- `frontend/styles.css`: estilos da interface.
- `frontend/app.js`: cliente HTTP para `/score/chat` com controle de conversa.

## 1. Create and activate virtual environment (`env`) on Windows

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.11 and rerun the command.

## 2. Train the model

```powershell
python training/train_model.py
```

This command reads `data/credit_dataset.csv`, trains a Logistic Regression model, and saves it to `models/credit_model.pkl`.

## 3. Run API locally

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:

- `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### Habilitar o endpoint conversacional com Gemini

O caminho mais simples para adicionar LLM neste projeto e usar o Gemini apenas
como camada de linguagem natural. O LLM extrai os campos do texto livre e o
modelo de regressao logistica continua sendo o unico motor real de score.

No PowerShell:

```powershell
$env:GEMINI_API_KEY="sua_chave_aqui"
$env:GEMINI_MODEL="gemini-2.0-flash"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Se `GEMINI_MODEL` nao for definido, a API usa `gemini-2.0-flash` por padrao.

## 4. Run with Docker

```powershell
docker compose up --build
```

The API will be exposed at `http://localhost:8000`.

Para usar Gemini com Docker, crie um arquivo `.env` na raiz do projeto com base
em `.env.example`:

```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.0-flash
```

## 5. Test `/score` endpoint using curl

PowerShell-friendly command:

```powershell
curl.exe -X POST "http://localhost:8000/score" -H "Content-Type: application/json" -d "{\"age\":35,\"income\":5000,\"number_of_loans\":2,\"payment_delays\":0}"
```

Expected response format:

```json
{
  "probability_default": 0.12
}
```

## 6. Test `/score/chat` endpoint using natural language

Esse endpoint usa o Gemini para extrair os campos do texto e depois chama o
mesmo modelo de score da API.

Agora ele tambem suporta memoria simples de conversa em processo:

- `conversation_id`: identifica a conversa.
- `turn`: numero do turno atual.
- Mensagens anteriores da mesma conversa sao usadas como contexto para extracao.

Exemplo com PowerShell:

```powershell
$body = @{
  message = "Tenho 35 anos, renda de 5000 por mes, 2 emprestimos e nenhum atraso."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/score/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Resposta esperada:

```json
{
  "conversation_id": "a1b2c3d4e5f6",
  "turn": 1,
  "extracted_data": {
    "age": 35,
    "income": 5000.0,
    "number_of_loans": 2,
    "payment_delays": 0
  },
  "missing_fields": [],
  "probability_default": 0.0549,
  "explanation": "O perfil foi convertido em dados estruturados e avaliado pelo modelo de regressao logistica."
}
```

Se o texto estiver incompleto, a API nao inventa valores. Ela devolve os campos
faltantes para o usuario complementar.

Perguntas de follow-up tambem sao suportadas. Exemplo:

```text
Usuario: Tenho 35 anos, renda de 5000 por mes, 2 emprestimos e nenhum atraso.
Usuario: Entao posso liberar credito?
```

Nesse caso, a API reutiliza o contexto da conversa e responde com uma orientacao
educacional baseada no ultimo score calculado, em vez de apenas repetir os
mesmos dados extraidos.

Exemplo de segunda mensagem usando a mesma conversa:

```json
{
  "message": "Tenho 2 emprestimos ativos e 1 atraso.",
  "conversation_id": "a1b2c3d4e5f6"
}
```

## 7. Rodar interface web minima (frontend)

Com a API rodando em `http://localhost:8000`, inicie um servidor estatico na
pasta `frontend`:

```powershell
cd frontend
python -m http.server 5500
```

Depois abra no navegador:

```text
http://localhost:5500
```

A interface envia mensagens para `/score/chat`, preserva o `conversation_id`
entre envios e permite resetar para iniciar nova conversa.

## Outros exemplos de payload

Perfil de baixo risco (cliente estável):

```json
{ "age": 45, "income": 7000, "number_of_loans": 1, "payment_delays": 0 }
```

Perfil de alto risco (jovem com muitos atrasos):

```json
{ "age": 22, "income": 1800, "number_of_loans": 3, "payment_delays": 5 }
```

---

## Diagrama de fluxo

```
CSV  →  train_model.py  →  StandardScaler + LogisticRegression  →  credit_model.pkl
                                                                          │
HTTP POST /score  ────────────────────────────────────────────────────────┘
         │ { age, income, number_of_loans, payment_delays }
         ↓
    predict_proba()
         ↓
    { "probability_default": 0.xx }
```

---

## Swagger UI

Com a API em execução (local ou Docker), acesse a documentação interativa em:

```
http://localhost:8000/docs
```

E possivel testar os endpoints `/score` e `/score/chat` diretamente pelo navegador.
