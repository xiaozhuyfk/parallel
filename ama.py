from __future__ import print_function

import logging
import globals
from relation_matching import modules
from sparql_backend import backend_util

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)




def answer(question):
    print(modules.facts_ranker.rank(question))

def test(dataset):
    print("test")

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


if __name__ == '__main__':
    main()