import json
import urllib.request
import urllib.error
from .config import settings

def chat_with_ollama(messages: list) -> str:
    """Envia o contexto e as mensagens para o Ollama local e retorna a resposta de texto."""
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(settings.OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get('message', {}).get('content', "Erro ao formatar resposta.")
    except urllib.error.URLError as e:
        print(f"[ERRO] Falha ao comunicar com Ollama. Verifique se ele está rodando em {settings.OLLAMA_URL}. Detalhe: {e}")
        return "Desculpe, meu cérebro local (Ollama) parece estar offline no momento."
    except Exception as e:
        print(f"[ERRO] Erro inesperado do Ollama: {e}")
        return f"Erro inesperado conectando ao agente: {e}"