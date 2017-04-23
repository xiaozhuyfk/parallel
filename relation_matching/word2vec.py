from gensim import models
from gensim import matutils
from numpy import dot
import logging

logger = logging.getLogger(__name__)
MIN_WORD_SIMILARITY = 0.4

class Word2Vec(object):

    def __init__(self, model_fname):
        self.embeddings = models.Word2Vec.load_word2vec_format(model_fname, binary=True)
        #self.embeddings = models.Word2Vec.load(model_fname)

    @staticmethod
    def init_from_config(config_options):
        embeddings_model = config_options.get('Alignment',
                                          'word-embeddings')
        return Word2Vec(embeddings_model)

    def transform(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            #logger.debug("'%s' don't have a word vector" % word)
            return None

    def transform_seq(self, tokens):
        V = []
        for t in tokens:
            v = self.transform(t)
            if v is not None:
                V.append(v)
        return V

    def synonym_score(self, word_a, word_b):
        """
        Returns a synonym score for the provided words.
        If the two words are not considered a synonym
        0.0 is returned.
        :param word_a:
        :param word_b:
        :return:
        """
        similarity = self.similarity(word_a, word_b)
        if similarity > MIN_WORD_SIMILARITY:
            return similarity
        else:
            return 0.0

    def similarity(self, word_a, word_b):
        try:
            a_vector = self.embeddings[word_a]
            b_vector = self.embeddings[word_b]
            diff = dot(matutils.unitvec(a_vector), matutils.unitvec(b_vector))
            return diff
        except KeyError:
            logger.debug("'%s' or '%s' don't have a word vector" % (word_a,
                                                                    word_b))
            return 0.0

    def embedding_similarity(self, embed_a, embed_b):
        diff = dot(matutils.unitvec(embed_a), matutils.unitvec(embed_b))
        return diff