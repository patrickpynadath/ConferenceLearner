import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_nips_soup(year):
    url = 'https://papers.nips.cc/paper/' + year
    doc = requests.get(url)
    soup = BeautifulSoup(doc.content, 'html5lib')
    return soup


def get_abstract_urls(soup):
    urls = soup.find_all("a")
    abstract_urls = []
    for u in urls:
        if 'Abstract' in u['href']:
            abstract_urls.append(u)
    return abstract_urls


def get_nips_paper_abstract(url_ending):
    url = 'https://papers.nips.cc/' + url_ending
    doc = requests.get(url)
    soup = BeautifulSoup(doc.content, 'html5lib')
    paragraphs = soup.find_all('p')
    target_paragraph = max(paragraphs, key= lambda p : len(p.text))
    return target_paragraph.text


def get_proceeding_abstracts(year):
    soup = get_nips_soup(year)
    abstract_urls = get_abstract_urls(soup)
    progress_bar = tqdm(enumerate(abstract_urls), total = len(abstract_urls))
    proceeding_info = []
    for idx, a in progress_bar:
        title = a.text
        url_ending = a['href']
        abstract = get_nips_paper_abstract(url_ending)
        proceeding_info.append((title, abstract))
    return proceeding_info
