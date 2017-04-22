import modules
import numpy as np

class Question(object):

    def __init__(self, query):
        self.query = query
        self.embedding = None


class QuestionEncoder(object):

    @staticmethod
    def position_encoding(tokens):
        w2v = modules.w2v

        Q = None
        V = []
        d = 300
        for i in xrange(len(tokens)):
            token = tokens[i]
            v = w2v.transform(token)
            if v is not None:
                V.append(v)

        q = len(V)
        for i in xrange(len(V)):
            v = V[i]
            for j in xrange(d):
                l_ij = min(float((i+1)*d)/((j+1)*q), float((j+1)*q)/((i+1)*d))
                v[j] *= l_ij

            if Q is None:
                Q = v
            else:
                Q += v
        return Q
