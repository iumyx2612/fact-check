from .wiki_element import WikiElement


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