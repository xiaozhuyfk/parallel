#!flask/bin/env python

from flask import Flask, jsonify, make_response
from relation_matching import modules

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/RESTapi/questions/<string:q>', methods=['GET'])
def get_tasks(q):
    result = str(len(q))
    if len(result) == 0:
        abort(404)
    result = modules.facts_ranker.rank(question)
    return jsonify({'response': result})

@app.errorhandler(404)
def no_response(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')