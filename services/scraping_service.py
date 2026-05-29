import time
import requests
from bs4 import BeautifulSoup

# Configuração simples de Cache na memória
_market_cache = {
    "data": None,
    "timestamp": 0
}

CACHE_TTL_SECONDS = 3600  # Atualiza no máximo 1 vez por hora

def fetch_market_data():
    """
    Realiza o web scraping de cotações agrícolas.
    Utiliza cache para evitar múltiplas requisições.
    Retorna os dados em formato de dicionário.
    """
    global _market_cache
    now = time.time()
    
    # Verifica o Cache
    if _market_cache["data"] and (now - _market_cache["timestamp"] < CACHE_TTL_SECONDS):
        return _market_cache["data"]
        
    url = "https://www.cepea.esalq.usp.br/br/indicador/soja.aspx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    market_info = {
        "source": url,
        "status": "success",
        "commodity": "Soja",
        "price_brl": "Não disponível",
        "date": "Não disponível"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Levanta erro para HTTP 4xx/5xx
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # O CEPEA exibe os indicadores numa tabela (id="imagenet-indicador1")
        # Vamos tentar capturar a primeira linha da tabela de cotações
        table = soup.find("table", {"id": "imagenet-indicador1"})
        
        if table:
            # Pega o corpo da tabela e a primeira linha
            tbody = table.find("tbody")
            first_row = tbody.find("tr")
            if first_row:
                cols = first_row.find_all("td")
                if len(cols) >= 2:
                    date_str = cols[0].text.strip()
                    price_str = cols[1].text.strip()
                    market_info["date"] = date_str
                    market_info["price_brl"] = price_str
        else:
            # Caso o HTML mude ou a tabela não seja encontrada
            market_info["status"] = "html_parser_error"
            
    except requests.RequestException as e:
        print(f"[SCRAPING ERROR] Falha ao acessar {url}: {e}")
        market_info["status"] = "network_error"
        market_info["error_details"] = str(e)
    except Exception as e:
        print(f"[SCRAPING ERROR] Erro inesperado: {e}")
        market_info["status"] = "unexpected_error"
        market_info["error_details"] = str(e)
        
    # Se falhar totalmente, fornece um fallback razoável para a demonstração
    if market_info["price_brl"] == "Não disponível":
        market_info["status"] = "mock_fallback"
        market_info["price_brl"] = "132,50 (Estimativa)"
        market_info["date"] = time.strftime("%d/%m/%Y")
        
    # Atualiza o cache
    _market_cache["data"] = market_info
    _market_cache["timestamp"] = now
    
    print("[INFO] Dados de mercado atualizados:", market_info)
    
    return market_info

def get_market_context_string() -> str:
    """Retorna uma string formatada para ser injetada no contexto do LLM."""
    data = fetch_market_data()
    return f"Cotação da {data['commodity']}: R$ {data['price_brl']} (Referência: {data['date']} - Status: {data['status']})"