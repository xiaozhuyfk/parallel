from __future__ import print_function

import logging
import globals
import modules

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train_model(model_name):
    modules.facts_ranker.train_model(model_name)

def train(dataset):
    modules.facts_ranker.train(dataset)

def test(dataset):
    modules.facts_ranker.test(dataset)

def extract_wiki(dataset, idx):
    print(modules.wiki_url["m.06815z"])
    #modules.support_sentence_extractor.extract_support_sentence(idx)
    #modules.facts_ranker.extract_wiki_data(dataset, idx)
    #modules.facts_ranker.extract_wiki_data_with_index(dataset, idx)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train memory network')
    train_parser.add_argument('dataset',
                              help='The dataset to train.')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test memory network')
    test_parser.add_argument('dataset',
                             help='The dataset to test')
    test_parser.set_defaults(which='test')

    process_parser = subparsers.add_parser('model', help="Train model")
    process_parser.add_argument('dataset',
                                help='Training dataset')
    process_parser.add_argument('name',
                                help='Training model name')
    process_parser.set_defaults(which='model')

    wiki_parser = subparsers.add_parser('wiki', help="extract wiki data")
    wiki_parser.add_argument('dataset', help='dataset')
    wiki_parser.add_argument("idx", help="partition index")
    wiki_parser.set_defaults(which='wiki')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)
    elif args.which == 'model':
        train_model(args.name)
    elif args.which == 'wiki':
        extract_wiki(args.dataset, int(args.idx))


if __name__ == '__main__':
    main()