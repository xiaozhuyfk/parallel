import logging
import modules
import time
import os
from relation_matching.util import codecsDumpJson, codecsLoadJson


logger = logging.getLogger(__name__)

class FactExtractor(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.cache_dir = config_options.get('Cache', 'cache-dir')
        self.backend = modules.sparql_backend

        self.entities_with_alias_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?s ?name where {
            ?s fb:common.topic.alias ?o .
            FILTER (lcase(str(?o)) = "%s") .
            ?s fb:type.object.name ?name
        }
        '''

        self.name_by_id_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?o where {
            fb:%s fb:type.object.name ?o
        }
        '''

        self.facts_by_id_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?r ?o where {
            fb:%s ?r ?o
        }
        '''

    @staticmethod
    def init_from_config(config_options):
        return FactExtractor(config_options)

    def in_cache(self, mid):
        file_path = self.cache_dir + mid
        return os.path.isfile(file_path)

    def load_from_cache(self, mid):
        file_path = self.cache_dir + mid
        return codecsLoadJson(file_path)

    def store_fact_list(self, mid, d):
        file_path = self.cache_dir + mid
        return codecsDumpJson(file_path, d)

    def extract_fact_list_with_entity_linker(self, question):
        start_time = time.time()
        entities = modules.entity_linker.identify_entities(question)
        logger.info("Identified identities: %s", str([ie.entity.name for ie in entities]))
        duration = (time.time() - start_time) * 1000
        logger.info("Entity Linking time: %.2f ms." % duration)

        start_time = time.time()
        result = []
        for ie in entities:
            e = ie.entity
            s, s_name = e.id, e.name

            if self.in_cache(s):
                result.append(self.load_from_cache(s))
                continue

            score = ie.surface_score
            s_name_result = self.backend.query(self.name_by_id_query % s)
            if s_name_result == []:
                s_name = e.name
            else:
                s_name = s_name_result[0][0]

            relations = {}
            facts = self.backend.query(self.facts_by_id_query % s)
            for f in facts:
                r, o = f[0], f[1]

                if o.startswith('m.'):
                    o_name = self.backend.query(self.name_by_id_query % o)
                    # skip if the entity does not have a name in Freebase
                    if o_name == []:
                        continue
                    if r in relations:
                        rel = relations[r]
                        rel["objects"].append(o_name[0][0])
                        rel["oid"].append(o)
                    else:
                        relations[r] = {"objects" : [o_name[0][0]],
                                        "oid" : [o]}
                elif o.startswith('g.'):
                    subfacts = self.backend.query(self.facts_by_id_query % o)
                    for subf in subfacts:
                        subr, subo = subf[0], subf[1]
                        subr = r + "\n" + subr
                        if subo.startswith('m.'):
                            o_name = self.backend.query(self.name_by_id_query % subo)
                            if o_name == []:
                                continue
                            if subr in relations:
                                rel = relations[subr]
                                rel["objects"].append(o_name[0][0])
                                rel["oid"].append(subo)
                            else:
                                relations[subr] = {"objects" : [o_name[0][0]],
                                                   "oid" : [subo]}
                        elif o.startswith('g.'):
                            continue
                        else:
                            if subr in relations:
                                rel = relations[subr]
                                rel["objects"].append(subo)
                                rel["oid"].append("ATTRIBUTE")
                            else:
                                relations[subr] = {"objects" : [subo],
                                                   "oid" : ["ATTRIBUTE"]}
                else:
                    if r in relations:
                        rel = relations[r]
                        rel["objects"].append(o)
                        rel["oid"].append("ATTRIBUTE")
                    else:
                        relations[r] = {"objects" : [o],
                                           "oid" : ["ATTRIBUTE"]}
            d = {"subject" : s_name,
                 "sid" : s,
                 "score" : score,
                 "relations" : relations}
            result.append(d)
            self.store_fact_list(s, d)

        duration = (time.time() - start_time) * 1000
        logger.info("Facts extraction time: %.2f ms." % duration)
        return result
