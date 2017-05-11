from __future__ import print_function

import logging
import globals
import sys
import time
from relation_matching import modules
from relation_matching.evaluation import load_eval_queries
from relation_matching.util import writeFile


logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)




def answer(question):
    print(modules.facts_ranker.rank(question))

def play():
    while True:
        sys.stdout.write("enter question> ")
        sys.stdout.flush()
        question = sys.stdin.readline().strip()
        if not question: continue

        top5 = modules.facts_ranker.rank(question)

        if not top5:
            print("Sorry, I don't know the answer.")

        for candidate in top5:
            print(candidate.graph_str)

def test(dataset):
    queries = load_eval_queries(dataset)
    file_path = "/home/ubuntu/parallel/duration.txt"
    writeFile(file_path, "")
    for query in queries:
        question = query.utterance
        start_time = time.time()
        modules.facts_ranker.rank(question)
        duration = (time.time() - start_time) * 1000
        writeFile(file_path, str(duration) + '\n', 'a')



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    answer_parser = subparsers.add_parser('answer')
    answer_parser.add_argument('question')
    answer_parser.set_defaults(which='answer')

    play_parser = subparsers.add_parser('play')
    play_parser.set_defaults(which='play')

    test_parser = subparsers.add_parser('test', help='Test memory network')
    test_parser.add_argument('dataset', help='The dataset to test')
    test_parser.set_defaults(which='test')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'test':
        train(args.dataset)
    elif args.which == 'answer':
        answer(args.question)
    elif args.which == 'play':
        play()


if __name__ == '__main__':
    main()