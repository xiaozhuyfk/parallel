import logging
from util import (
    codecsReadFile
)

logger = logging.getLogger(__name__)

class WikiUrl(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.data = config_options.get('Wiki', 'url-path')
        self.wiki_name = {}
        prefix = "http://en.wikipedia.org/wiki/"

        with open(self.data) as f:
            for line in f:
                if line:
                    triple = line.strip().split("\t")
                    mid = triple[0][1:].replace("/", ".")
                    name = None
                    for url in triple[1:]:
                        suffix = url[len(prefix):]
                        if suffix.startswith("index.html?"):
                            continue
                        if "%" in suffix:
                            continue
                        name = suffix

                    if (name is not None and "/" in name):
                        name = name.replace("/", "|")
                    if (name is not None):
                        self.wiki_name[mid] = name
        logger.info("Done loading wiki urls.")

    @staticmethod
    def init_from_config(config_options):
        return WikiExtractor(config_options)

    def __getitem__(self, item):
        return self.wiki_name.get(item)



if __name__ == '__main__':
    with open("../test_result/result.txt") as f:
        for line in f:
            print [line]