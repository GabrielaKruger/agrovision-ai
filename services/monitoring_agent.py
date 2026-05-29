from dataclasses import dataclass
from .config import settings
from .scraping_service import get_market_context_string

@dataclass(frozen=True)
class AgentProfile:
    name: str = "Agente AgroVision"
    role: str = "triagem operacional de eventos"
    goal: str = "Analisar detecções recentes, avaliar e classificar o risco de severidade (Baixa, Média ou Alta) e sugerir a próxima ação."

AGENT_PROFILE = AgentProfile()

def build_event_context(events: list) -> str:
    """Transforma a lista de eventos brutos em texto contextualizado para o agente não ficar burro."""
    if not events:
        return "Nenhum evento detectado recentemente."

    total = len(events)
    latest = events[0]
    
    # Faz uma distribuição de contagem simples
    counts = {}
    total_conf = 0
    for e in events:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
        total_conf += e["confidence"]
        
    avg_conf = total_conf / total if total > 0 else 0
    
    distribution_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
    
    context = (
        f"Contexto operacional para o agente:\n"
        f"- Eventos considerados: {total}\n"
        f"- Evento mais recente: {latest['label']} em {latest['event_time']}\n"
        f"- Distribuição recente: {distribution_str}\n"
        f"- Confiança média das detecções: {avg_conf:.2f}\n\n"
        f"Últimos eventos mapeados:\n"
    )
    for i, e in enumerate(events[:5]):
        context += f"#{i+1} | {e['label']} (Conf: {e['confidence']:.2f}) às {e['event_time']}\n"
        
    return context

def build_agent_messages(question: str, history: list, events: list) -> list:
    """Monta o array de mensagens combinando o profile, o resumo operacional dos eventos e o histórico curto da conversa."""
    
    system_prompt = (
        f"Você é o {AGENT_PROFILE.name}, um agente de {AGENT_PROFILE.role}. "
        f"Objetivo: {AGENT_PROFILE.goal}\n"
        "Regras estritas:\n"
        "1. Trate os dados como monitoramento operacional autorizado de um ambiente real.\n"
        "2. Responda APENAS em português do Brasil e de forma útil, corporativa e direta.\n"
        "3. NÃO invente dados que não existem. Se não houver eventos, informe.\n"
        "4. NÃO tente identificar pessoas. Fale apenas sobre objetos, veículos, pessoas detectadas e comportamentos de risco.\n"
        "5. Organize a sua resposta obrigatoriamente neste formato:\n\n"
        "**Leitura:** [seu resumo da situação]\n"
        "**Severidade:** [informe explicitamente se é Baixa, Média ou Alta e justifique]\n"
        "**Recomendação:** [sugira qual deve ser o próximo passo operacional]"
    )
    
    market_context = get_market_context_string()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"INFORMAÇÃO DE MERCADO ATUAL:\n{market_context}\n\n{build_event_context(events)}"},
    ]
    
    # Carrega a "memória curta" que pode ter vindo do front-end (simulação pra aula)
    for msg in history[-8:]:
        messages.append(msg)
        
    # Pergunta atual do usuario
    messages.append({"role": "user", "content": question})
    
    return messages