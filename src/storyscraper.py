#!/usr/bin/env python

from bs4 import BeautifulSoup
from requests import get
from tqdm.auto import tqdm
import pandas as pd

url = "https://americanliterature.com/short-story-library"
base_url = "https://americanliterature.com/"

css_selector = ".col-md-4 > a"

response = get(url)
soup = BeautifulSoup(response.text, "html.parser")


def list_flatten(l, a=None):
    if a is None:
        a = []
    for i in l:
        if isinstance(i, list):
            list_flatten(i, a)
        else:
            a.append(i)
    return a


def scrape_page_urls(pagenum):
    url = f"https://americanliterature.com/short-story-library?page={pagenum}"
    response = get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return [scrape_story(link["href"]) for link in tqdm(soup.select(css_selector), desc="2nd loop")]


def scrape_story(rel_url):
    resp = get(base_url + rel_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs_list = [p.contents for p in soup.select("p")][:-6]  # This removes extra tags at end
    paragraphs_list = [p for p in paragraphs_list if len(p) > 0]
    paragraphs_list = [[t if isinstance(t, str) else '\n\n' for t in p] + ['\n'] for p in paragraphs_list]
    return " ".join([str(t) for t in list_flatten(paragraphs_list)])


story_list = [scrape_page_urls(page) for page in tqdm(range(1, 16), desc="1st loop")]

df = pd.DataFrame({'text': list_flatten(story_list)})
df.text = df.text.str.replace("<br/>", "\n\n")
df.to_csv("../data/external/scraped_stories.csv", index=False)
print("Finished")
