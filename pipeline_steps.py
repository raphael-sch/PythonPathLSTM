from model import LSTMModel
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
from text_types import Frame
from feature_set import FeatureSet
from utils import Config


class PipelineStep(object):

    def __init__(self, parsed_text, step_name, action='pred', output_path='./output', trained_model=None):
        if not parsed_text.gold and action == 'train':
            raise ValueError('Need gold labels in parsed text')
        self.parsed_text = parsed_text
        self.step_name = step_name
        self.action = action
        self.output_path = output_path
        self.trained_model = trained_model
        self.actions = dict(train=self._execute_train, pred=self._execute_pred)

    def print(self, s, pos_type=None):
        if pos_type:
            print(self.step_name + ' - ' + pos_type + ': ' + s)
        else:
            print(self.step_name + ': ' + s)

    def execute(self):
        return self.actions[self.action]()

    def _execute_train(self):
        args = ['N', 'V']
        #with Pool() as p: results = p.map(self._execute_train_mp, args)
        results = [self._execute_train_mp(a) for a in args]
        trained_model_step = dict()
        for pos_type, trained_model_step_pos in zip(args, results):
            trained_model_step[pos_type] = trained_model_step_pos
        return trained_model_step

    def _execute_pred(self):
        pass


class PredicateIdentification(PipelineStep):

    def __init__(self, parsed_text, action='pred', output_path='./output', trained_model=None):
        super().__init__(parsed_text, 'pi', action, output_path, trained_model)

    def _execute_train(self):
        pass

    def _execute_pred(self):
        for frame_gold, sentence in self._yield_frames():
            fram_pred = Frame(frame_gold.frame_id, frame_gold.pred, frame_gold.pred_word_id, frame_gold.pos_type)
            sentence.add_frame_pred(fram_pred)

    def _yield_frames(self, gold=True):
        get_frames = lambda s: s.get_frames_gold() if gold else s.get_frames_pred()
        for sentence in self.parsed_text:
            for frame in get_frames(sentence):
                yield frame, sentence


class ArgumentIdentificationStep(PipelineStep):

    def __init__(self, parsed_text, action='pred', output_path='./output', trained_model=None):
        super().__init__(parsed_text, 'ai', action, output_path, trained_model)
        if not trained_model and action == 'pred':
            raise ValueError('Need trained model to make predictions')

    def _execute_train_mp(self, pos_type):
        config = Config(self.step_name + '_' + pos_type)
        print(config)
        iterator = self._get_pred_word_iterator(pos_type, True)
        feature_set = FeatureSet(iterator, self.step_name, pos_type, config, freezed=False, label_func=self._get_labels,
                                 vocabs=None, class_names=None)

        model_name = self.parsed_text.input_name + '_ai_' + pos_type + '_'
        model = LSTMModel(model_name, self.output_path)
        model.train(feature_set, config)

        trained_model_step_pos = dict()
        trained_model_step_pos['model'] = model
        trained_model_step_pos['config'] = config

        trained_model_step_pos['class_names'] = feature_set.class_names

        vocabs = feature_set.get_vocabs()
        trained_model_step_pos['vocabs'] = vocabs
        return trained_model_step_pos

    def _execute_pred(self):
        lstm_model_path = self.trained_model['lstm_model_path']
        for pos_type in ['V', 'N']:
            saved_model_step_pos = self.trained_model[pos_type]
            config = saved_model_step_pos['config']
            config.set_value('lstm_model_path', lstm_model_path)
            print(config)
            iterator = self._get_pred_word_iterator(pos_type, gold=False)

            feature_set = FeatureSet(iterator, self.step_name, pos_type, config, freezed=True, label_func=None,
                                     vocabs=saved_model_step_pos['vocabs'],
                                     class_names=saved_model_step_pos['class_names'])
            model = saved_model_step_pos['model']
            y_pred = model.pred(feature_set, config)

            # write predicted argument words to text object
            for y, pred_word_pair in zip(y_pred, iterator()):
                pred_word, word, frame_predicted, sentence = pred_word_pair
                if y == 'Arg':
                    frame_predicted.add_arg('Arg', word.word_id)

    @staticmethod
    def _get_labels(iterator):
        label_array = list()
        for pred_word, word, frame_gold, sentence in iterator():
            args_word_ids = frame_gold.args.keys()
            label = 'NoArg'
            if word.word_id in args_word_ids:
                label = 'Arg'
            label_array.append(label)
        return label_array

    def _get_pred_word_iterator(self, pos_type, gold):
        def get_frames(s):
            return s.get_frames_gold() if gold else s.get_frames_pred()

        def iterator():
            for sentence in self.parsed_text:
                for frame in get_frames(sentence):
                    if frame.pos_type != pos_type:
                        continue
                    pred_word_id = frame.pred_word_id
                    pred_word = sentence.get_word(pred_word_id)
                    for word in sentence:
                        yield pred_word, word, frame, sentence
        return iterator


class ArgumentClassificationStep(PipelineStep):

    def __init__(self, parsed_text, action='pred', output_path='./output', trained_model=None):
        super().__init__(parsed_text, 'ac', action, output_path, trained_model)
        if not trained_model and action == 'pred':
            raise ValueError('Need trained model to make predictions')

    def _execute_train_mp(self, pos_type):
        config = Config(self.step_name + '_' + pos_type)
        print(config)
        iterator = self._get_pred_word_iterator(pos_type, True)
        feature_set = FeatureSet(iterator, self.step_name, pos_type, config, freezed=False, label_func=self._get_labels,
                                 vocabs=None, class_names=None)

        model_name = self.parsed_text.input_name + '_ac_' + pos_type + '_'
        model = LSTMModel(model_name, self.output_path)
        model.train(feature_set, config)

        trained_model_step_pos = dict()
        trained_model_step_pos['model'] = model
        trained_model_step_pos['config'] = config

        trained_model_step_pos['class_names'] = feature_set.class_names

        vocabs = feature_set.get_vocabs()
        trained_model_step_pos['vocabs'] = vocabs
        return trained_model_step_pos

    def _execute_pred(self):
        lstm_model_path = self.trained_model['lstm_model_path']
        for pos_type in ['V', 'N']:
            saved_model_step_pos = self.trained_model[pos_type]
            config = saved_model_step_pos['config']
            config.set_value('lstm_model_path', lstm_model_path)
            print(config)
            iterator = self._get_pred_word_iterator(pos_type, gold=False)

            feature_set = FeatureSet(iterator, self.step_name, pos_type, config, freezed=True, label_func=None,
                                     vocabs=saved_model_step_pos['vocabs'],
                                     class_names=saved_model_step_pos['class_names'])

            model = saved_model_step_pos['model']
            y_pred = model.pred(feature_set, config)

            # write predicted argument types to text object
            for y, pred_word_pair in zip(y_pred, iterator()):
                pred_word, arg_word, frame_predicted, sentence = pred_word_pair
                frame_predicted.add_arg(y, arg_word.word_id)

    @staticmethod
    def _get_labels(iterator):
        label_array = list()
        for pred_word, arg_word, frame_gold, sentence in iterator():
            args = frame_gold.args
            label_array.append(args[arg_word.word_id])
        return np.asarray(label_array)

    def _get_pred_word_iterator(self, pos_type, gold):
        def get_frames(s):
            return s.get_frames_gold() if gold else s.get_frames_pred()

        def iterator():
            for sentence in self.parsed_text:
                for frame in get_frames(sentence):
                    if frame.pos_type != pos_type:
                        continue
                    pred_word_id = frame.pred_word_id
                    pred_word = sentence.get_word(pred_word_id)
                    for word_id, arg in frame.args.items():
                        arg_word = sentence.get_word(word_id)
                        yield pred_word, arg_word, frame, sentence
        return iterator

available_steps = dict(pi=PredicateIdentification, ai=ArgumentIdentificationStep, ac=ArgumentClassificationStep)
