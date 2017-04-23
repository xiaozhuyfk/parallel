import logging

logger = logging.getLogger(__name__)

def get_id_with_url(backend, url):
    query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?s where {
            fb:m.0100s3s6 fb:common.topic.topical_webpage ?s
        }
    '''

    mid = backend.query(query)
    return mid
