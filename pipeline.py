import pickle
from pipeline_steps import available_steps
from features import essential_vocab_features
from scorer import ConLL2009Scorer
from concurrent.futures import ProcessPoolExecutor as Pool


class Pipeline(object):

    def __init__(self, parsed_text, action, output_path, trained_model):
        """
        Base class for the different pipelines.
        :param parsed_text: text_types.ParsedText
        :param action: train, test or pred
        :param output_path:
        :param trained_model: all necessary information will be written to this dict, in order to reload a trained model
        """
        self.parsed_text = parsed_text
        self.action = action
        self.output_path = output_path
        self.trained_model = trained_model
        self.steps = ['pi', 'ai', 'ac']
        self.name = self.parsed_text.input_name + '_' + '_'.join(self.steps)

    def start(self):
        pass

    def _finish(self):
        pass

    def _step_finished(self, step_name, step):
        pass


class TrainPipeline(Pipeline):

    def __init__(self, parsed_text, output_path='./output/'):
        """
        Initializes Pipeline to train models based on the gold-labels in parsed_text. Writes model files into output_path.
        :param parsed_text: used to train the model
        :param output_path: model will be written to this
        """
        super().__init__(parsed_text, 'train', output_path, dict())
        if not self.parsed_text.gold:
            raise ValueError('In order to train, parsed text needs to have annotations (labels)')
        # restore vocabularies of trained model in order to have the same one-hot-vectors
        for v_name, vocab in parsed_text.essential_vocabs.items():
            essential_vocab_features[v_name].vocab = vocab

        self.trained_model['essential_vocabs'] = parsed_text.essential_vocabs

    def start(self):
        """
        Starts the different steps and passes the model_object to them
        :return:
        """
        # mp should work, not tested with tensorflow
        #with Pool() as p: results = p.map(self.start_mp, self.steps)
        results = [self.start_mp(s) for s in self.steps]
        for step_name, trained_model_step in zip(self.steps, results):
            self.trained_model[step_name] = trained_model_step
        return self._finish()

    def start_mp(self, step_name):
        print(step_name + ': starting step')
        step = available_steps[step_name](self.parsed_text, 'train', self.output_path, None)
        step_model_object = step.execute()
        print(step_name + ': finished step')
        return step_model_object

    def _finish(self):
        """
        Calles after the pipeline processed all steps
        :return: path to the model_object
        """
        model_file_path = self._save_trained_model(self.trained_model)
        return model_file_path

    def _save_trained_model(self, trained_model):
        name = self.name + '.model'
        model_file_path = self.output_path + name
        with open(model_file_path, 'wb') as f:
            pickle.dump(trained_model, f)
        print('wrote model {} to {}'.format(name, self.output_path))
        return model_file_path


class PredPipeline(Pipeline):

    def __init__(self, parsed_text, trained_model_file, output_path='./output/'):
        """
        Prediction Pipeline.
        :param parsed_text: holds the sentences to be predicted
        :param trained_model_file: the trained model
        :param output_path: where to write the *.PRED file
        """
        trained_model = pickle.load(open(trained_model_file, 'rb'))
        lstm_model_path = '/'.join(trained_model_file.split('/')[:-1]) + '/'
        trained_model['lstm_model_path'] = lstm_model_path
        super().__init__(parsed_text, 'pred', output_path, trained_model)
        for v_name, vocab in trained_model['essential_vocabs'].items():
            essential_vocab_features[v_name].vocab = vocab

    def start(self):
        """
        Starts the different steps and passes the model_object to them
        :return:
        """
        for step_name in self.steps:
            print(step_name + ': starting step')
            train_model_step = self.trained_model[step_name]
            if train_model_step:
                train_model_step['lstm_model_path'] = self.trained_model['lstm_model_path']
            step = available_steps[step_name](self.parsed_text, 'pred', self.output_path, train_model_step)
            step.execute()
            self._step_finished(step_name, step)
            print(step_name + ': finished step')
        return self._finish()

    def _finish(self):
        filename = self.name + '.PRED'
        pred_file_path = self.parsed_text.write(filename, self.output_path, 'conll2009', gold=False)
        return pred_file_path


class TestPipeline(Pipeline):

    def __init__(self, parsed_text_test, parsed_text_gold, trained_model_file, output_path='./output/'):
        """
        Testing Pipeline. Uses the Prediction Pipeline to get predictions on the parsed_text. Evaluates against
        the gold labels with a CoNLL Scorer. Writes *.RESULTS to output_path
        :param parsed_text: holds the sentences with gold labels to be tested
        :param trained_model_file: the trained model which is used to predict the new labels
        :param output_path: where to write the *.RESULTS, *.PRED, *.GOLD file
        """
        super().__init__(parsed_text_test, 'test', output_path, trained_model_file)
        if not parsed_text_gold.gold:
            raise ValueError('In order to test, parsed text needs to have annotations (labels)')
        self.parsed_text_gold = parsed_text_gold
        self.pred_pipeline = PredPipeline(self.parsed_text, trained_model_file, self.output_path)
        self.scorer = ConLL2009Scorer()

    def start(self):
        """
        Use Prediction Pipeline to predict new labels. Evaluate new labels against gold labels with CoNLL 2009 scorer.
        :return: path to *.RESULT file
        """
        pred_file_path = self.pred_pipeline.start()

        gold_filename = self.name + '.GOLD'
        gold_file_path = self.parsed_text_gold.write(gold_filename, self.output_path, 'conll2009', gold=True)

        results_filename = self.name + '.RESULTS'

        scorer = ConLL2009Scorer(pred_file_path, gold_file_path, self.output_path, results_filename)
        print('start running evaluation script: {}'.format(scorer.perl_script))
        result_file_path = scorer.run()
        return result_file_path
