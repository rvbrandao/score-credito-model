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
- Docker

## Project Structure

```text
project-root/
  data/
    credit_dataset.csv
  training/
    train_model.py
  api/
    main.py
    model_loader.py
    schemas.py
  models/
    credit_model.pkl
  requirements.txt
  Dockerfile
  docker-compose.yml
  README.md
```

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

## 4. Run with Docker

```powershell
docker compose up --build
```

The API will be exposed at `http://localhost:8000`.

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

É possível testar o endpoint `/score` diretamente pelo navegador, sem precisar do curl.
