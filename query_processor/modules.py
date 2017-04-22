import logging
import globals
from word2vec import Word2Vec
from fact_extractor import FactExtractor
from wiki_extractor import WikiAPIExtractor, WikiExtractor
from wiki_url import WikiUrl
from corenlp_parser.parser import CoreNLPParser
from entity_linker.entity_linker import EntityLinker
from ranker import Ranker

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None
extractor = None
parser = None
entity_linker = None
facts_ranker = None
wiki_extractor = None
support_sentence_extractor = None
wiki_url = None

def init_from_config(args):
    global w2v, sparql_backend, extractor, parser, entity_linker, facts_ranker, wiki_extractor, support_sentence_extractor
    global wiki_url
    config_options = globals.config

    #w2v = Word2Vec.init_from_config(config_options)
    #sparql_backend = globals.get_sparql_backend(config_options)
    extractor = FactExtractor.init_from_config(args, config_options)
    #parser = CoreNLPParser.init_from_config()
    #entity_linker = EntityLinker.init_from_config()
    facts_ranker = Ranker.init_from_config(config_options)
    #wiki_extractor = WikiAPIExtractor.init_from_config(config_options)
    support_sentence_extractor = WikiExtractor.init_from_config(config_options)
    wiki_url = WikiUrl(config_options)
