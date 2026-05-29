# 🌾 AgroVision AI

# Relatório de Revisão Arquitetural, Segurança e Implementação de Web Scraping

## Integrantes

* Gabriela Krüger

---

# 1. Introdução

O AgroVision AI é um sistema de monitoramento inteligente voltado ao ambiente rural, desenvolvido com o objetivo de detectar pessoas, veículos e movimentações utilizando Visão Computacional através do modelo YOLO11.

O sistema utiliza FastAPI para disponibilização da API, SQLite para persistência dos eventos detectados, YOLO11 para detecção de objetos, Ollama para interpretação contextual dos eventos e uma camada de Web Scraping para enriquecimento das informações apresentadas ao usuário.

O objetivo deste relatório é apresentar a análise arquitetural, avaliação de segurança, melhorias realizadas sobre o código gerado por IA e a implementação da camada de Web Scraping.

---

# 2. Parte 1 — Revisão da Arquitetura

## Estrutura Geral

```text
agrovision_ia/
├── app.py
├── templates/
├── static/
├── services/
│   ├── config.py
│   ├── event_repository.py
│   ├── monitoring_agent.py
│   ├── ollama_client.py
│   ├── scraping_service.py
│   ├── schemas.py
│   └── video_monitor.py
├── detections.db
├── uploads/
├── runs/
├── dataset_agro/
└── models/
```

---

## Frontend

### Componentes

```text
templates/index.html
static/
```

### Responsabilidades

* Exibir câmera monitorada
* Exibir eventos detectados
* Exibir evidências capturadas
* Exibir respostas da IA
* Atualizar informações em tempo real

### Avaliação

A interface atua apenas como camada de apresentação dos dados.

Toda a lógica principal encontra-se no backend.

**Resultado:** Arquitetura adequada.

---

## Backend / API

### Arquivo Principal

```text
app.py
```

### Responsabilidades

* Disponibilização das rotas
* Comunicação com serviços internos
* Processamento das requisições
* Integração com banco de dados
* Integração com IA
* Integração com Web Scraping

### Avaliação

Toda a lógica principal do sistema está concentrada no backend.

**Resultado:** Adequado.

---

## Banco de Dados

### Tecnologia

```text
SQLite
```

### Arquivo

```text
detections.db
```

### Informações Armazenadas

* ID da detecção
* Tipo do objeto detectado
* Data e horário
* Confiança da detecção
* Evidência capturada

### Avaliação

O acesso ao banco está separado da interface.

Arquivo responsável:

```text
services/event_repository.py
```

**Resultado:** Adequado.

---

## Camada de Visão Computacional

### Tecnologia

```text
YOLO11
```

### Responsabilidades

* Captura dos frames da câmera
* Processamento das imagens
* Identificação de pessoas
* Identificação de veículos
* Geração de eventos
* Armazenamento das evidências

### Arquivo Principal

```text
services/video_monitor.py
```

### Avaliação

A camada de visão computacional encontra-se desacoplada da interface e do banco de dados.

Isso permite substituir futuramente o modelo YOLO por outra solução sem grandes alterações no sistema.

**Resultado:** Adequado.

---

## Camada de Inteligência Artificial

### Tecnologias

```text
YOLO11
Ollama
```

### Responsabilidades

* Detecção de objetos
* Interpretação dos eventos
* Geração de respostas em linguagem natural

### Arquivos

```text
services/video_monitor.py
services/ollama_client.py
```

### Avaliação

A IA encontra-se separada da lógica principal da API.

**Resultado:** Adequado.

---

## Agente Inteligente (Ollama)

### Tecnologia

```text
Ollama
```

### Endpoint

```text
http://127.0.0.1:11434/api/chat
```

### Responsabilidades

* Receber contexto dos eventos detectados
* Interpretar informações armazenadas
* Gerar respostas para o usuário
* Auxiliar na análise dos eventos

### Exemplos

* Quantas pessoas foram detectadas?
* Qual foi o último evento registrado?
* Houve movimentação suspeita?
* Quais eventos ocorreram nas últimas horas?

### Arquivo

```text
services/ollama_client.py
```

### Avaliação

A integração externa está isolada em um serviço próprio.

**Resultado:** Adequado.

---

## Camada de Web Scraping

### Arquivo

```text
services/scraping_service.py
```

### Responsabilidades

* Coleta de previsão do tempo
* Coleta de alertas climáticos
* Consulta de notícias do agronegócio
* Consulta de informações públicas relevantes

### Avaliação

A camada foi implementada separadamente da API e da interface.

**Resultado:** Adequado.

---

## Fluxo Geral do Sistema

```text
Câmera
   ↓
YOLO11
   ↓
Detecção de Objetos
   ↓
Banco SQLite
   ↓
API FastAPI
   ↓
Dashboard Web
   ↓
Usuário

           ↓

        Ollama
           ↓

Interpretação dos Eventos

           ↓

     Web Scraping
           ↓

 Informações Climáticas
 Notícias do Agronegócio
 Alertas e Contexto
```

---

## Conclusão da Arquitetura

| Item                           | Status |
| ------------------------------ | ------ |
| Frontend separado              | ✅      |
| Backend separado               | ✅      |
| Banco separado                 | ✅      |
| Serviços internos separados    | ✅      |
| YOLO separado                  | ✅      |
| Ollama separado                | ✅      |
| Integrações externas separadas | ✅      |
| Web Scraping separado          | ✅      |

---

# 3. Parte 2 — Revisão de Segurança

## Variáveis de Ambiente

O sistema utiliza configurações centralizadas através do arquivo `.env`.

### Exemplos

```env
OLLAMA_URL
OLLAMA_MODEL
MODEL_PATH
CAMERA_SOURCE
CONFIDENCE_THRESHOLD
```

### Benefícios

* Evita hardcoding de configurações
* Facilita manutenção
* Melhora segurança

---

## Validação de Entradas

Os dados recebidos pela API passam por validação antes do processamento.

### Exemplo

```python
if not message:
    return "Mensagem inválida"
```

### Benefícios

* Evita entradas vazias
* Reduz falhas inesperadas
* Melhora a estabilidade

---

## Proteção Contra SQL Injection

O sistema utiliza consultas parametrizadas.

### Exemplo

```python
cursor.execute(
    "SELECT * FROM events WHERE id=?",
    (event_id,)
)
```

### Benefícios

* Redução do risco de SQL Injection
* Maior segurança no acesso ao banco

---

## Tratamento de Erros

Os serviços implementam tratamento de exceções.

### Exemplo

```python
try:
    ...
except Exception:
    ...
```

### Benefícios

* Maior estabilidade
* Menor exposição de detalhes internos
* Melhor experiência do usuário

---

## Segurança no Processamento de Arquivos

O sistema realiza validação dos arquivos processados.

### Benefícios

* Redução de arquivos inválidos
* Maior segurança operacional
* Menor risco de falhas

---

## Segurança do Web Scraping

A camada de scraping possui:

* Tratamento de erros
* Controle de requisições
* Cache temporário
* Validação dos dados coletados

### Benefícios

* Evita sobrecarga da fonte
* Melhora desempenho
* Aumenta confiabilidade

---

## Conclusão da Segurança

| Item                   | Status |
| ---------------------- | ------ |
| Variáveis de ambiente  | ✅      |
| Validação de entrada   | ✅      |
| Proteção SQL Injection | ✅      |
| Tratamento de erros    | ✅      |
| Segurança de upload    | ✅      |
| Segurança do scraping  | ✅      |

---

# 4. Parte 3 — Melhorias do Código Gerado por IA

## Melhoria 1 — Centralização das Configurações

### Código Original

Configurações distribuídas em diversos arquivos.

### Problema

Dificuldade de manutenção e atualização.

### Solução

Criação do arquivo:

```text
services/config.py
```

### Benefício

Centralização das configurações do sistema.

---

## Melhoria 2 — Tratamento de Falhas do Ollama

### Código Original

Não existia tratamento adequado para falhas de conexão.

### Problema

O sistema poderia interromper sua execução caso o Ollama estivesse indisponível.

### Solução

```python
except urllib.error.URLError:
    return "Desculpe, meu cérebro local parece estar offline."
```

### Benefício

Maior robustez e estabilidade.

---

## Melhoria 3 — Separação dos Serviços

### Código Original

Grande parte da lógica concentrada em poucos arquivos.

### Problema

Alto acoplamento e baixa escalabilidade.

### Solução

Criação da estrutura:

```text
services/
```

### Benefício

Maior organização e manutenção.

---

## Melhoria 4 — Criação da Camada de Web Scraping

### Código Original

O sistema não possuía dados externos para contextualização.

### Problema

As detecções eram apresentadas sem informações complementares.

### Solução

Implementação de:

```text
services/scraping_service.py
```

### Benefício

Maior valor informacional para o usuário.

---

## Melhoria 5 — Organização da Camada de IA

### Código Original

A lógica de IA encontrava-se fortemente acoplada ao fluxo principal.

### Problema

Dificuldade para manutenção.

### Solução

Separação entre:

```text
services/video_monitor.py
services/ollama_client.py
```

### Benefício

Maior modularidade.

---

# 5. Parte 4 — Implementação da Camada de Web Scraping

## Objetivo

Complementar os eventos detectados com informações externas relevantes ao ambiente rural.

---

## Dados Coletados

O sistema realiza coleta de:

* Previsão do tempo
* Temperatura atual
* Umidade relativa do ar
* Alertas climáticos
* Notícias do agronegócio
* Informações públicas do setor agrícola

---

## Justificativa

As condições climáticas possuem influência direta sobre:

* Segurança da propriedade
* Operações agrícolas
* Movimentação de máquinas
* Atividades dos trabalhadores

Os dados coletados complementam as informações obtidas pelo YOLO e auxiliam na tomada de decisão.

---

## Serviço Implementado

### Arquivo

```text
services/scraping_service.py
```

### Responsabilidades

* Buscar dados externos
* Organizar dados em JSON
* Disponibilizar informações ao sistema

---

## Tratamento de Erros

```python
try:
    ...
except Exception:
    return {
        "status": "indisponivel"
    }
```

---

## Controle de Requisições

Foi implementado mecanismo de cache e limite de consultas.

### Objetivos

* Evitar excesso de requisições
* Melhorar desempenho
* Reduzir consumo de recursos

---

## Estrutura dos Dados

```json
{
  "cidade": "Toledo",
  "temperatura": 24,
  "umidade": 68,
  "clima": "Ensolarado",
  "alerta": "Nenhum alerta ativo"
}
```

---

## Integração ao Sistema

Os dados coletados são utilizados:

* No dashboard principal
* Nas respostas do Ollama
* No contexto dos eventos detectados
* Nas análises realizadas pelo sistema

---

# Conclusão

O AgroVision AI apresenta uma arquitetura organizada, com separação clara entre frontend, backend, banco de dados, serviços internos, inteligência artificial, visão computacional e integração externa.

A revisão identificou melhorias aplicadas ao código originalmente gerado por IA, resultando em maior organização, segurança, manutenção e escalabilidade.

A implementação da camada de Web Scraping agrega informações contextuais relevantes ao ambiente rural monitorado, tornando o sistema mais completo e útil para apoio à tomada de decisão.
