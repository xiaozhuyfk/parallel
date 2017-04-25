#!flask/bin/env python

from flask import Flask, jsonify, make_response

"""
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
"""

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/ama/<string:q>', methods=['GET'])
def get_tasks(q):
    result = str(len(q))
    if len(result) == 0:
        abort(404)
    #top5 = modules.facts_ranker.rank(question)
    #result = [candidate.graph_str for candidate in top5]
    return jsonify({'response': result})

@app.errorhandler(404)
def no_response(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')