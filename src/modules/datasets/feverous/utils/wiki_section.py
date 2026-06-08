from .wiki_element import WikiElement
from .feveous_utils import wiki_links_to_md_links


class WikiSection(WikiElement):
    def __init__(self, name, section, page):
        self.content = section["value"]
        self.level = section["level"]
        self.name = name
        self.page = page

    def id_repr(self):
        return self.name

    def get_id(self):
        return self.name

    def get_ids(self):
        return [self.name]

    def __str__(self):
        return self.content

    def get_level(self):
        return self.level


class MDSection(WikiSection):
    def __str__(self):
        return "#" * self.level + " " + wiki_links_to_md_links(self.content)