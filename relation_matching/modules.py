import logging
import globals
from relation_matching.wiki_url import WikiUrl
from relation_matching.fact_extractor import FactExtractor
from entity_linking.entity_linker import EntityLinker
from word2vec import Word2Vec

from ranker import Ranker

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None
entity_linker = None
facts_extractor = None
facts_ranker = None
wiki_url = None

def init_from_config(args):
    global w2v, sparql_backend, entity_linker, facts_ranker, facts_extractor
    global wiki_url
    config_options = globals.config

    w2v = Word2Vec.init_from_config(config_options)
    sparql_backend = globals.get_sparql_backend(config_options)
    wiki_url = WikiUrl(config_options)
    entity_linker = EntityLinker.init_from_config(config_options, wiki_url)
    facts_ranker = Ranker.init_from_config(config_options)
    facts_extractor = FactExtractor.init_from_config(config_options)
