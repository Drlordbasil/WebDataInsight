import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class QueryGenerator:
    def __init__(self, topics):
        self.topics = topics

    def generate_search_queries(self):
        search_queries = []
        for topic in self.topics:
            search_queries.append(f"site:example.com {topic}")
        return search_queries


class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    def scrape_urls(self, search_queries):
        urls = []
        for query in search_queries:
            search_url = f"https://www.google.com/search?q={query}"
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/url?q='):
                    url = href[7:].split('&')[0]
                    urls.append(url)
        return urls

    def scrape_content(self, urls):
        scraped_data = []
        for url in urls:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            images = [img['src'] for img in soup.find_all('img')]
            links = [a['href'] for a in soup.find_all('a')]
            scraped_data.append({
                'url': url,
                'text': text,
                'images': images,
                'links': links
            })
        return scraped_data


class ContentAnalyzer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = pipeline(self.model_name)

    def analyze_sentiment(self, text):
        sentiment = self.model('sentiment-analysis', text)[0]
        return sentiment['label']

    def analyze_summary(self, text):
        summary = self.model('summarization', text)[0]
        return summary['summary_text']

    def analyze_keywords(self, text):
        keywords = self.model('ner', text)
        return [entity['word'] for entity in keywords]

    def analyze_entities(self, text):
        entities = self.model('ner', text)
        return [entity['entity'] for entity in entities]


Base = declarative_base()


class ScrapedData(Base):
    __tablename__ = 'scraped_data'
    id = Column(Integer, primary_key=True)
    url = Column(String(200))
    text = Column(Text)
    images = Column(String(500))
    links = Column(String(500))


class DataStorage:
    def __init__(self, db_name):
        self.db_name = db_name
        self.engine = create_engine(f'sqlite:///{self.db_name}.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def store_data(self, scraped_data):
        session = self.Session()
        for data in scraped_data:
            data_to_store = ScrapedData(
                url=data['url'],
                text=data['text'],
                images=', '.join(data['images']),
                links=', '.join(data['links'])
            )
            session.add(data_to_store)
        session.commit()

    def retrieve_data(self):
        session = self.Session()
        data = session.query(ScrapedData).all()
        return [{
            'url': d.url,
            'text': d.text,
            'images': d.images.split(', '),
            'links': d.links.split(', ')
        } for d in data]

    def generate_word_cloud(self, text):
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()


class AutonomousWebContentScraperAndAnalyzer:
    def __init__(self, topics, db_name):
        self.topics = topics
        self.db_name = db_name
        self.query_generator = QueryGenerator(self.topics)
        self.web_scraper = WebScraper()
        self.content_analyzer = ContentAnalyzer('bert-base-uncased')
        self.data_storage = DataStorage(self.db_name)

    def scrape_and_analyze(self):
        search_queries = self.query_generator.generate_search_queries()
        urls = self.web_scraper.scrape_urls(search_queries)
        scraped_data = self.web_scraper.scrape_content(urls)
        self.data_storage.store_data(scraped_data)

        data = self.data_storage.retrieve_data()
        texts = [d['text'] for d in data]
        combined_text = ' '.join(texts)
        self.data_storage.generate_word_cloud(combined_text)

        sentiment_scores = []
        summaries = []
        for d in data:
            sentiment = self.content_analyzer.analyze_sentiment(d['text'])
            sentiment_scores.append(sentiment)
            summary = self.content_analyzer.analyze_summary(d['text'])
            summaries.append(summary)

        keywords = self.content_analyzer.analyze_keywords(combined_text)
        entities = self.content_analyzer.analyze_entities(combined_text)

        df = pd.DataFrame(
            {'Sentiment': sentiment_scores, 'Summary': summaries})
        print(df.head())
        print("Keywords:", keywords)
        print("Entities:", entities)


if __name__ == "__main__":
    topics = ['technology', 'business']
    db_name = 'web_content'

    scraper = AutonomousWebContentScraperAndAnalyzer(topics, db_name)
    scraper.scrape_and_analyze()
