from text_types import Frame, Sentence, ParsedText, Word


class Parser(object):

    def __init__(self, pattern, language='eng', gold=False):
        """
        Base class for different CoNLL parsers. Takes a pattern for the different fields and extracts the most important
        ones.
        """
        self.pattern = pattern
        self.language = language
        self.gold = gold
        self.element_count = dict(form=dict(), lemma=dict(), gpos=dict(), deprel=dict())
        self.min_count = 2 # values with less than min_count will be replaced by '__DUMMY__' in the vocabulary

    def get_parsed_text(self, filepath):
        """
        Parse CoNLL format files and return a parsed_text object from it.
        :param filepath: the file to be parsed
        :return: parsed_text object
        """
        filename = filepath.split('/')[-1]
        parsed_text = ParsedText(input_name=filename, gold=self.gold)
        with open(filepath, 'r', encoding='utf-8') as f:
            sentence_split_lines = list()
            sentence_id = 0
            for line in f:
                line = line.rstrip().split('\t')
                if len(line) > 1:
                    line = list(map(lambda l: self._placeholder_to_none(l), line))
                    elements = {e_name: line[self.pattern[e_name]] for e_name in
                                ['word_id', 'form', 'lemma', 'gpos', 'head', 'deprel']}
                    elements['word_id'], elements['head'] = int(elements['word_id']), int(elements['head'])
                    if self.gold:
                        for v_name, vocab in self.element_count.items():
                            vocab[elements[v_name]] = vocab.get(elements[v_name], 0) + 1
                    pred = line[self.pattern['pred']]
                    args = line[self.pattern['args']:]
                    sentence_split_lines.append((elements, pred, args))
                else:
                    sentence = Sentence(sentence_id)
                    sentence_id += 1
                    frame_id = 0
                    # create a frame for every predicate
                    for elements, pred, args in sentence_split_lines:
                        if pred:
                            frame = Frame(frame_id, pred, elements['word_id'], elements['gpos'])
                            sentence.add_frame_gold(frame)
                            frame_id += 1
                    # put argument (by position) to the correct frame
                    for elements, pred, args in sentence_split_lines:
                        word = Word(**elements)
                        sentence.add_word(word)
                        for frame_id, arg in enumerate(args):
                            if arg:
                                sentence.get_frame_gold(frame_id).add_arg(arg, elements['word_id'])
                    sentence_split_lines = list()
                    parsed_text.add_sentence(sentence)
        if self.gold:
            essential_vocabs = {v_name: dict(__DUMMY__=0) for v_name in self.element_count.keys()}
            for v_name, element_counts in self.element_count.items():
                for element, count in element_counts.items():
                    if count >= self.min_count:
                        essential_vocabs[v_name][element] = len(essential_vocabs[v_name])
            parsed_text.essential_vocabs = essential_vocabs
        # connects the children in the sentences
        parsed_text.finish()
        return parsed_text

    @staticmethod
    def _placeholder_to_none(s):
        return None if s == '_' else s


class ConLL2008Parser(Parser):

    def __init__(self, language=None, gold=False):
        pattern = dict(word_id=0, form=1, lemma=2, gpos=3, ppos=4, split_form=5, split_lemma=6, pposs=7, head=8,
                       deprel=9, pred=10, args=11)
        super().__init__(pattern, language, gold)


class ConLL2009Parser(Parser):

    def __init__(self, language=None, gold=False):
        pattern = dict(word_id=0, form=1, lemma=2, plemma=3, gpos=4, ppos=5, feat=6, pfeat=7, head=8, phead=9,
                       deprel=10, pdeprel=11, fillpred=12, pred=13, args=14)
        super().__init__(pattern, language, gold)


def get_parser(parser_format, gold=False, language='eng'):
    if parser_format == 'conll2008':
        return ConLL2008Parser(language, gold)
    elif parser_format == 'conll2009':
        return ConLL2009Parser(language, gold)
    else:
        raise AttributeError('No parser with format: {} found'.format(parser_format))