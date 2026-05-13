# Relatório do Projeto AgroVision

## Parte 1 — Revisão da Arquitetura

### Análise Inicial
O projeto original misturava responsabilidades: o backend continha lógica de banco de dados, IA e processamento de stream em um único arquivo. O frontend tinha lógica de atualização em JavaScript, mas era simples.

### Melhorias Implementadas
- **Estrutura de Pastas**:
  - `src/backend/`: Contém `app.py` com rotas e lógica principal.
  - `src/database/`: `database.py` para acesso ao banco.
  - `src/services/`: `ai_service.py` para IA, `scraping_service.py` para scraping.
  - `src/frontend/`: `templates/` para HTML, `static/` para assets.
- **Frontend**: Mantido simples, apenas exibição. Lógica de atualização movida para scripts, mas sem regras de negócio.
- **Backend/API**: Centralizado em FastAPI, com rotas protegidas por autenticação.
- **Banco de Dados**: Isolado em `src/database/database.py`, com funções dedicadas para acesso.
- **Serviços Internos**: Processamento de stream separado no backend. IA em `src/services/ai_service.py`.
- **Camada de IA/Modelo**: Separada em módulo `ai_service.py`, isolando chamadas ao YOLO.
- **Camada de Integração Externa**: Nova camada de web scraping em `src/services/scraping_service.py`, integrada via APIs.

A arquitetura agora suporta crescimento, com separação clara de responsabilidades.

## Parte 2 — Revisão de Segurança

### Riscos Identificados
- APIs abertas sem autenticação.
- Dados não validados.
- Possível exposição de erros técnicos.
- Risco de processamento de dados maliciosos via scraping.

### Melhorias
- Adicionada autenticação básica HTTP em todas as rotas usando `HTTPBasic`.
- Senhas movidas para variáveis de ambiente (`ADMIN_USERNAME`, `ADMIN_PASSWORD`).
- Tratamento de erros com HTTPExceptions, evitando exposição de detalhes internos.
- Validação implícita via dependências do FastAPI para tipos de dados.
- Scraping com cache (1 hora) e limites para evitar sobrecarga e ataques.

## Parte 3 — Melhoria do Código Gerado com IA

### Trecho Original: Funções de Banco de Dados
**Original**: Código inline em app.py, conexões diretas ao SQLite.
**Problema**: Acesso ao DB espalhado, sem isolamento, risco de SQL Injection se não usar placeholders.
**Melhoria**: Movido para `src/database/database.py`, com funções dedicadas usando placeholders.
**Por que melhor**: Separação de responsabilidades, facilita manutenção, testes e segurança.

### Trecho Original: Processamento de IA
**Original**: Modelo YOLO carregado globalmente, lógica inline no loop de stream.
**Problema**: Misturado com processamento de stream, difícil de testar e reutilizar.
**Melhoria**: Função `detect_objects` em `src/services/ai_service.py`, chamada no backend.
**Por que melhor**: Isolamento da IA, permite reutilização, testes unitários e manutenção.

### Trecho Original: Scraping
**Original**: Mock simples para notícias.
**Problema**: Não fornece dados reais, limitado.
**Melhoria**: Scraping real de RSS do Google News para agricultura, com cache e tratamento de erro.
**Por que melhor**: Dados reais e relevantes, com boas práticas de scraping.

### Trecho Original: Frontend
**Original**: Lógica de fetch em script, sem autenticação no cliente.
**Problema**: Sem validação, poderia ter regras indevidas, exposição de endpoints.
**Melhoria**: Mantido simples, apenas exibição; autenticação no backend.
**Por que melhor**: Frontend focado em UI, backend em lógica e segurança.

## Parte 4 — Implementação de uma Camada de Web Scraping

### Justificativa
Para AgroVision, que detecta movimentações agrícolas (pessoas, veículos, animais), dados como previsão do tempo, notícias agrícolas e preços de commodities enriquecem o sistema, permitindo alertas baseados em condições climáticas, notícias relevantes ou flutuações de preços.

### Implementação
- **Fonte**: wttr.in para tempo (pública, gratuita), RSS do Google News para notícias agrícolas, mock para preços de commodities (devido a restrições de scraping de sites financeiros).
- **Tratamento de Erro**: Try-except com timeouts, cache para limitar requisições.
- **Limite**: Cache de 1 hora para tempo e notícias, 1 hora para commodities.
- **Integração**: Endpoints `/weather`, `/news`, `/commodities`, exibidos no frontend via JavaScript.
- **Estrutura de Dados**: JSON estruturado com campos relevantes.

### Relevância
Melhora o projeto ao fornecer contexto ambiental e econômico para detecções, como alertar sobre chuva durante movimentações ou preços baixos afetando operações.

## Como Executar
1. Instalar dependências: `pip install -r requirements.txt`
2. Rodar: `uvicorn src.backend.app:app --reload`
3. Acessar: http://localhost:8000
4. Usar login:
   - usuário: `admin`
   - senha: `password`

## Link do Git
[GitHub Repository](https://github.com/GabrielaKruger/agrovision-ai.git) 







