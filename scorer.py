import subprocess


class Scorer(object):

    def __init__(self, file_pred, file_gold, perl_script, output_path, output_filename):
        self.file_pred = file_pred
        self.file_gold = file_gold
        self.perl_script = perl_script
        self.output_path = output_path
        self.output_filename = output_filename
        self.evaluation = None

    def run(self):
        result_file_path = self.output_path + self.output_filename
        exec = ['perl', self.perl_script, '-g ' + str(self.file_gold), '-s ' + str(self.file_pred)]
        pipe = subprocess.Popen(' '.join(exec), stdout=subprocess.PIPE, shell=True)
        evaluation = pipe.communicate()[0].decode('utf-8')
        with open(result_file_path, 'w') as f:
            f.write(evaluation)
        self.evaluation = evaluation
        print('Wrote results to {}'.format(result_file_path))
        return result_file_path


class ConLL2009Scorer(Scorer):

    def __init__(self, file_pred=None, file_gold=None, output_path='./output/', output_filename='RESULTS'):
        perl_script = './score_scripts/eval09.pl'
        super().__init__(file_pred, file_gold, perl_script, output_path, output_filename)

    def run(self):
        result_file_path = super().run()
        evaluation = self.evaluation.split('\n')
        # write semantic score to sys out
        for line in evaluation[5:18]:
            print(line)
        return result_file_path


class ConLL2008Scorer(Scorer):

    def __init__(self, file_pred=None, file_gold=None, output_path='./output/', output_filename='RESULTS'):
        perl_script = './score_scripts/eval08.pl'
        super().__init__(file_pred, file_gold, perl_script, output_path, output_filename)

    def run(self):
        result_file_path = super().run()
        evaluation = self.evaluation.split('\n')
        # don't know the lines for semantic score in scorer 2008
        for line in evaluation:
            print(line)
        return result_file_path
