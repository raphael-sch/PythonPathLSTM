class Frame(object):

    def __init__(self, frame_id, pred, pred_word_id, gpos):
        self.frame_id = frame_id
        self.pred = pred
        self.pos_type = Frame._get_pos_type(gpos)
        self.pred_word_id = pred_word_id
        self.args = dict()

    @staticmethod
    def _get_pos_type(gpos):
        if gpos in ['VBG', 'VBN', 'VBD', 'VBP', 'VBZ', 'V']:
            return 'V'
        return 'N'

    def add_arg(self, arg, word_id):
        self.args[word_id] = arg

    def __repr__(self):
        return ', '.join(str(e) for e in [self.frame_id, self.pred, self.pred_word_id, self.pos_type])


class ParsedText(object):

    def __init__(self, sentences=None, input_name=None, gold=False, essential_vocabs=None):
        self.sentences = dict()
        self.input_name = input_name
        self.gold = gold
        self.essential_vocabs = essential_vocabs
        if sentences:
            self.sentences = {s.sentence_id: s for s in sentences}

    def add_sentence(self, sentence):
        self.sentences[sentence.sentence_id] = sentence

    def get_sentence(self, sentence_id):
        return self.sentences[sentence_id]

    def get_subset(self, factor):
        new_len = int(len(self) / float(factor))
        return ParsedText([self.sentences[i] for i in range(new_len)], self.input_name + '.subset' + str(factor),
                          self.gold, self.essential_vocabs)

    def __getitem__(self, item):
        if item not in self.sentences:
            raise IndexError
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

    def finish(self):
        for sentence in self:
            sentence.connect_children()

    def write(self, filename, output_path='./output/', format='conll2009', gold=False):
        get_frames = lambda s: s.get_frames_gold() if gold else s.get_frames_pred()
        with open(output_path + filename, 'w', encoding='utf-8') as f:
            if format == 'conll2008':
                for sentence in self:
                    frames = get_frames(sentence)
                    word_id_frame = {frame.pred_word_id: frame for frame in frames}
                    for word in sentence:
                        s = ''
                        word_id = word.word_id
                        for element in ['word_id', 'form', 'lemma', 'gpos', 'ppos', 'split_form',
                                        'split_lemma','pposs', 'head', 'deprel']:
                            value = word.get_element(element)
                            s += str(value) + '\t'
                        if word_id in word_id_frame:
                            s += word_id_frame[word_id].pred + '\t'
                        else:
                            s += '_\t'
                        for frame in frames:
                            if word_id in frame.args:
                                s += frame.args[word_id] + '\t'
                            else:
                                s += '_\t'
                        s = s[:-1] + '\n'
                        f.write(s)
                    f.write('\n')
            if format == 'conll2009':
                for sentence in self:
                    frames = get_frames(sentence)
                    word_id_frame = {frame.pred_word_id: frame for frame in frames}
                    for word in sentence:
                        s = ''
                        word_id_intern = word.word_id
                        word_id, form, lemma, gpos, ppos, split_form, split_lemma, pposs, head, deprel = \
                            [word.get_element(element) for element in ['word_id', 'form', 'lemma', 'gpos', 'ppos', 'split_form',
                                        'split_lemma','pposs', 'head', 'deprel']]
                        s += str(word_id) + '\t' + str(form) + '\t' + str(lemma) + '\t' + str('_') + '\t' + str(gpos) + '\t' + str(ppos) + '\t' + str('_') + '\t' + str('_') + '\t' + str(head) + '\t' + str('_') + '\t' + str(deprel) + '\t' + str('_') + '\t'
                        if word_id_intern in word_id_frame:
                            s += 'Y' + '\t'
                            s += word_id_frame[word_id_intern].pred + '\t'
                        else:
                            s += '_\t'
                            s += '_\t'
                        for frame in frames:
                            if word_id_intern in frame.args:
                                s += frame.args[word_id_intern] + '\t'
                            else:
                                s += '_\t'
                        s = s[:-1] + '\n'
                        f.write(s)
                    f.write('\n')
        return output_path + filename


class Sentence(object):

    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.root = None
        self.words = dict()
        self.frames_gold = dict()
        self.frames_pred = dict()
        self.enable_cache = False

    def add_word(self, word):
        if word.deprel == 'ROOT':
            self.root = word
        self.words[word.word_id] = word

    def get_word(self, word_id):
        return self.words[word_id]

    def add_frame_gold(self, frame):
        self.frames_gold[frame.frame_id] = frame

    def get_frame_gold(self, frame_id):
        return self.frames_gold[frame_id]

    def get_frames_gold(self):
        return [frame[1] for frame in sorted(self.frames_gold.items())]

    def add_frame_pred(self, frame):
        self.frames_pred[frame.frame_id] = frame

    def get_frame_pred(self, frame_id):
        return self.frames_pred[frame_id]

    def get_frames_pred(self):
        return [frame[1] for frame in sorted(self.frames_pred.items())]

    def get_rightmost_dep_word(self, word, cache=dict()):
        if word in cache:
            return cache[word]
        for right_word_id in reversed(range(word.word_id +1, len(self)+1)):
            right_word = self.get_word(right_word_id)
            if right_word in word.direct_children:
                if self.enable_cache:
                    cache[word] = right_word
                return right_word
        if self.enable_cache:
            cache[word] = None
        return None

    def get_leftmost_dep_word(self, word, cache=dict()):
        if word in cache:
            return cache[word]
        for left_word_id in range(1, word.word_id):
            left_word = self.get_word(left_word_id)
            if left_word in word.direct_children:
                if self.enable_cache:
                    cache[word] = left_word
                return left_word
        if self.enable_cache:
            cache[word] = None
        return None

    def get_right_sibling_word(self, word, cache=dict()):
        if word in cache:
            return cache[word]
        if self.enable_cache:
            cache[word] = None
        if word.word_id == self.root.word_id:
            return None
        head_word = self.get_word(word.head)
        for right_word_id in range(word.word_id + 1, len(self)+1):
            right_word = self.get_word(right_word_id)
            if right_word in head_word.direct_children:
                if self.enable_cache:
                    cache[word] = right_word
                return right_word
        return None

    def get_left_sibling_word(self, word, cache=dict()):
        if word in cache:
            return cache[word]
        if self.enable_cache:
            cache[word] = None
        if word.word_id == self.root.word_id:
            return None
        head_word = self.get_word(word.head)
        for left_word_id in reversed(range(1, word.word_id)):
            left_word = self.get_word(left_word_id)
            if left_word in head_word.direct_children:
                if self.enable_cache:
                    cache[word] = left_word
                return left_word
        return None

    def get_dep_path(self, word_1, word_2, cache=dict()):
        """from word_1 to word_2"""
        if self.enable_cache:
            cache_key = (word_1, word_2)
            if cache_key in cache:
                return cache[cache_key]

        cur_word = word_2
        word_2_to_root = dict()
        idx = 0
        while cur_word is not self.root:
            # if circle: break
            if cur_word.word_id in word_2_to_root:
                break
            word_2_to_root[cur_word.word_id] = idx
            cur_word = self.get_word(cur_word.head)
            idx += 1
        word_2_to_root[self.root.word_id] = idx

        word_1_to_word_2 = []
        cur_word = word_1
        while cur_word.word_id not in word_2_to_root:
            word_1_to_word_2.append(cur_word)
            cur_word = self.get_word(cur_word.head)
        intersection = word_2_to_root[cur_word.word_id]
        common_idx = len(word_1_to_word_2)

        word_2_to_root = {idx: word_id for word_id, idx in word_2_to_root.items()}
        for i in reversed(range(0, intersection+1)):
            word_1_to_word_2.append(self.get_word(word_2_to_root[i]))
        if self.enable_cache:
            cache[cache_key] = word_1_to_word_2
        return word_1_to_word_2, common_idx

    def connect_children(self):
        for word in self:
            if word is self.root:
                continue
            self.get_word(word.head).add_child(word)
        self.root.collect_children()

    def __getitem__(self, item):
        item += 1
        if item not in self.words:
            raise IndexError
        return self.words[item]

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return ' '.join([str(self[i]) for i in range(len(self.words))])


class Word(object):

    def __init__(self, word_id=None, form=None, lemma=None, gpos=None, head=None, deprel=None):
        self.word_id = word_id
        self.form = form
        self.lemma = lemma
        self.gpos = gpos
        self.head = head
        self.deprel = deprel
        self.direct_children = set()
        self.children = set()

    def get_element(self, name, placeholder='_'):
        attr = getattr(self, name, None)
        if attr is None:
            attr = placeholder
        return attr

    def add_child(self, child_word):
        self.direct_children.add(child_word)

    def collect_children(self):
        for child in self.direct_children:
            self.children.update(child.collect_children())
        return self.children

    def __repr__(self):
        return str(self.form)