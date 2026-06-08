from .wiki_element import WikiElement, process_text
from .feveous_utils import wiki_links_to_md_links


class WikiSentence(WikiElement):
    def __init__(self, name, sentence, page):
        self.name = name
        self.content = process_text(sentence)
        self.page = page

    def id_repr(self):
        return self.name

    def get_id(self):
        return self.name

    def __str__(self):
        return self.content

    def get_ids(self):
        return [self.name]


class MDSentence(WikiSentence):
    def __str__(self):
        return wiki_links_to_md_links(self.content)