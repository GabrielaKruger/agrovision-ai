import requests
from bs4 import BeautifulSoup
import time
import json

# Cache to avoid too many requests
cache = {}
cache_time = {}

def get_weather_data(location="São Paulo, Brazil"):
    now = time.time()
    if location in cache and now - cache_time.get(location, 0) < 3600:  # 1 hour cache
        return cache[location]

    try:
        url = f"https://wttr.in/{location.replace(' ', '+')}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather = {
            'location': location,
            'temperature': data['current_condition'][0]['temp_C'],
            'humidity': data['current_condition'][0]['humidity'],
            'description': data['current_condition'][0]['weatherDesc'][0]['value'],
            'wind_speed': data['current_condition'][0]['windspeedKmph']
        }
        cache[location] = weather
        cache_time[location] = now
        return weather
    except Exception as e:
        return {'error': str(e)}

def get_agricultural_news():
    now = time.time()
    if 'news' in cache and now - cache_time.get('news', 0) < 3600:  # 1 hour cache
        return cache['news']

    try:
        url = "https://news.google.com/rss/search?q=agricultura+brasil&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')[:5]  # Limit to 5 news
        news = []
        for item in items:
            title = item.title.text if item.title else 'No title'
            link = item.link.text if item.link else ''
            pub_date = item.pubDate.text if item.pubDate else ''
            news.append({'title': title, 'link': link, 'date': pub_date})
        result = {'news': news}
        cache['news'] = result
        cache_time['news'] = now
        return result
    except Exception as e:
        return {'error': str(e)}

def get_commodity_prices():
    # Mock for commodity prices, as scraping real sites may violate terms
    # In real implementation, scrape from public sources like BMF or similar
    return {
        'commodities': [
            {'name': 'Milho', 'price': 'R$ 85,00/saca', 'date': '2023-10-01'},
            {'name': 'Soja', 'price': 'R$ 180,00/saca', 'date': '2023-10-01'}
        ]
    }