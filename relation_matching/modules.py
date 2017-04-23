import logging
import globals
from word2vec import Word2Vec

from ranker import Ranker

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None
entity_linker = None
facts_ranker = None

def init_from_config(args):
    global w2v, sparql_backend, entity_linker, facts_ranker
    global wiki_url
    config_options = globals.config

    #w2v = Word2Vec.init_from_config(config_options)
    sparql_backend = globals.get_sparql_backend(config_options)
    #entity_linker = EntityLinker.init_from_config()
    #facts_ranker = Ranker.init_from_config(config_options)
