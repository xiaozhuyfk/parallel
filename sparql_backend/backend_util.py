import logging

logger = logging.getLogger(__name__)

def get_id_with_url(backend, url):
    query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?s where {
            ?s fb:common.topic.topical_webpage %s
        }
    '''

    mid = backend.query(self.facts_by_id_query % url)
    return mid
