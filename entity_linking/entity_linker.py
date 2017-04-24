import logging
import tagme


tagme.GCUBE_TOKEN = "38345877-16db-4293-887f-8a9154c95247-843339462"

logger = logging.getLogger(__name__)

class Entity(object):

    def __init__(self, name):
        self.name = name


class KBEntity(Entity):

    def __init__(self, name, identifier, score):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.id = identifier
        # A popularity score.
        self.score = score


class IdentifiedEntity():

    def __init__(self,
                 name,
                 entity,
                 surface_score = 0):
        # A readable name to be displayed to the user.
        self.name = name
        # A score for the match of those tokens.
        self.surface_score = surface_score
        # The identified entity object.
        self.entity = entity


class EntityLinker(object):

    def __init__(self, config_options, wiki_url):
        self.config_options = config_options
        self.wiki_url = wiki_url

    @staticmethod
    def init_from_config(config_options, wiki_url):
        return EntityLinker(config_options, wiki_url)

    def identify_entities(self, text):
        result = []
        annotations = tagme.annotate(text)
        for ann in annotations.get_annotations(0.1):
            name = ann.entity_title
            score = ann.score
            wiki_title = tagme.normalize_title(name)
            logger.info("Wiki title: " + wiki_title)
            mid = self.wiki_url[wiki_title]
            if mid is None: continue

            e = KBEntity(name, mid, score)
            ie = IdentifiedEntity(name, e, score)
            result.append(ie)

        return result




def main():
    pass

if __name__ == '__main__':
    main()