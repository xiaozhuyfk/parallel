#!flask/bin/env python

from flask import Flask, jsonify, make_response
from flask_cors import CORS, cross_origin
from OpenSSL import SSL
import os

from relation_matching import modules
import globals

import argparse
parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

parser.add_argument('--config',
                    default='config.cfg',
                    help='The configuration file to use')
args = parser.parse_args()

# Read global config
globals.read_configuration(args.config)

# Load modules
modules.init_from_config(args)



#context = SSL.Context(SSL.SSLv23_METHOD)
cer = os.path.join(os.path.dirname(__file__), 'certs/development.crt')
key = os.path.join(os.path.dirname(__file__), 'certs/development.key')
context = (cer, key)

application = Flask(__name__)
app = application
CORS(app)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/ama/<string:q>', methods=['GET'])
def get_tasks(q):
    result = str(len(q))
    if len(result) == 0:
        abort(404)
    top5 = modules.facts_ranker.rank(q)
    result = [candidate.graph_str for candidate in top5]
    return jsonify({'response': result})

@app.route('/network/<string:q>', methods=["GET"])
def network(q):
    result = modules.facts_ranker.network(q)
    return jsonify(result)

@app.errorhandler(404)
def no_response(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=443, ssl_context=context)
