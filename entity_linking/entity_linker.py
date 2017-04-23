import logging
import requests

logger = logging.getLogger(__name__)


class EntityLinker(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.api_url = "https://tagme.d4science.org/tagme/api"
        self.tag_url = "http://tagme.d4science.org/tagme/tag"
        self.spot_url = "https://tagme.d4science.org/tagme/spot"
        self.key = "38345877-16db-4293-887f-8a9154c95247-843339462"

    @staticmethod
    def init_from_config(config_options):
        return EntityLinker(config_options)

    def tagme_tagging(self,
                      text,
                      lang = "en",
                      tweet = "false",
                      include_abstract = "false",
                      include_categories = "false",
                      include_all_spots = "false",
                      long_text = 0,
                      epsilon = 0.3):
        parameter = {
            'gcube-token' : self.key,
            'text' : text,
            'lang' : lang,
            'tweet' : tweet,
            'include_abstract' : include_abstract,
            'include_categories' : include_categories,
            'long_text' : long_text,
            'epsilon' : epsilon
        }

        r = requests.get(self.tag_url, data = parameter)
        annotations = r.content

        return annotations




def main():
    print EntityLinker(None).tagme_tagging("what was procter & gamble 's net profit in 1955")

if __name__ == '__main__':
    main()