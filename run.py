from parser import get_parser
from pipeline import TrainPipeline, PredPipeline, TestPipeline
from utils import check_path
import random
import argparse
import sys

random.seed(1337)

if sys.version_info[0] != 3 or sys.version_info[1] < 4 or sys.version_info[2] < 3:
    print("This script requires at least Python version 3.4.3")
    #sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run Neural Semantic Role Labeling')
    parser.add_argument('action', choices=['train', 'test', 'predict'])
    parser.add_argument('input_file', type=argparse.FileType('r'))
    parser.add_argument('file_format', default='conll2009', choices=['conll2008', 'conll2009'])
    parser.add_argument('output_folder', nargs='?', default='./output/', type=check_path)
    parser.add_argument('model_file', nargs='?', type=argparse.FileType('r'))
    args = parser.parse_args()

    if args.action != 'train' and args.model_file is None:
        raise AttributeError('Need a trined model to predict or test')

    action = args.action
    input_file = args.input_file.name
    file_format = args.file_format
    output_folder = args.output_folder

    if action == 'train':
        train(input_file, file_format, output_folder)
    if action == 'test':
        model_file = args.model_file.name
        test(input_file, file_format, output_folder, model_file)
    if action == 'predict':
        model_file = args.model_file.name
        predict(input_file, file_format, output_folder, model_file)


def train(input_file_gold, input_file_format, output_path):
    """
    Reads the input_file into a parsed_text object to forward this to the training pipeline.
    The training pipeline earns a model from the gold labels of input file. Writes trained model to output_path.
    :param input_file_gold: CoNLL annotated input file with gold labels
    :param input_file_format: type of CoNLL annotation: 'conll2008' or 'conll2009'
    :param output_path: model will be written into this folder
    :return: writes trained model into output_path
    """
    parser_gold = get_parser(input_file_format, gold=True, language='eng')
    parsed_text_gold = parser_gold.get_parsed_text(input_file_gold)
    pipeline = TrainPipeline(parsed_text_gold, output_path)
    pipeline.start()


def test(input_file, input_file_format, output_path, trained_model_file):
    """
    Reads the input_file into a parsed_text object to forward this to the testing pipeline.
    The testing pipeline uses the trained model to predict predicates and arguments in input_file and
    evaluates them with the CoNLL-Scorer against the gold labels of input_file.
    Writes a *.RESULTS *.PRED and *.GOLD to output_path
    :param input_file: CoNLL annotated input file with gold labels
    :param input_file_format: type of CoNLL annotation: 'conll2008' or 'conll2009'
    :param output_path: model will be written into this folder
    :param trained_model_file: trained model from training step (TrainPipeline)
    :return: writes output of CoNLL-Scorer (*.RESULTS), *.PRED, *.GOLD to output_path
    """
    parser_gold = get_parser(input_file_format, gold=True, language='eng')
    parsed_text_gold = parser_gold.get_parsed_text(input_file)

    parser_input = get_parser(input_file_format, gold=False, language='eng')
    parsed_text_test = parser_input.get_parsed_text(input_file)

    pipeline = TestPipeline(parsed_text_test, parsed_text_gold, trained_model_file, output_path)
    pipeline.start()


def predict(input_file, input_file_format, output_path, trained_model_file):
    """
    Reads the input_file into a parsed_text object to forward this to the predicting pipeline.
    The predicting pipeline uses the trained model to predict predicates and arguments in input_file and writes
    them in a *.PRED file to output_path.
    :param input_file: CoNLL annotated input file (no gold labels needed)
    :param input_file_format: type of CoNLL annotation: 'conll2008' or 'conll2009'
    :param output_path: model will be written into this folder
    :param trained_model_file: trained model from training step (TrainPipeline)
    :return: writes predicted predicated and arguments (*.PRED) to output_path
    """
    parser = get_parser(input_file_format, gold=True, language='eng')
    parsed_text = parser.get_parsed_text(input_file)

    pipeline = PredPipeline(parsed_text, trained_model_file, output_path)
    pipeline.start()


if __name__ == '__main__':
    main()
