class Feature(object):

    def __init__(self, name=None):
        self.name = name
        if name is None:
            self.name = type(self).__name__

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")

    def get_idx(self, pred_word, arg_word, frame, sentence, freezed):
        raise NotImplementedError("Please Implement this method")

    def get_vector_batch(self, iterator, freezed=True):
        feature_vectors = list()
        for pred_word, arg_word, frame, sentence in iterator():
            feature_idx = self.get_idx(pred_word, arg_word, frame, sentence, freezed)
            feature_vectors.append({feature_idx: 1})
        return feature_vectors


class ElementMixin(object):

    def get_element_value(self, word):
        if word is None:
            return None
        return word.get_element(self.element)


class EssentialVocabFeature(Feature):

    def __init__(self, name=None):
        super().__init__(name)

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")

    def get_idx(self, pred_word, arg_word, frame, sentence, freezed=False):
        value = self.get_value(pred_word, arg_word, frame, sentence)
        idx = self.get_idx_by_value(value)
        return idx

    def get_idx_by_value(self, value):
        idx = self.vocab[value] if value in self.vocab else self.vocab['__DUMMY__']
        return idx

    def __len__(self):
        return len(self.vocab)


class UniqueVocabFeature(Feature):

    def __init__(self, min_count=1, name=None):
        super().__init__(name)
        self.vocab = dict()
        self.vocab_name = self.name + '_vocab'
        self.min_count = min_count

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")

    def get_idx(self, pred_word, arg_word, frame, sentence, freezed):
        value = self.get_value(pred_word, arg_word, frame, sentence)
        if not freezed and len(self.vocab) == 0:
            raise ValueError('Can\'t return index if vocab is not loaded')
        idx = self.vocab[value] if value in self.vocab else self.vocab['__DUMMY__']
        return idx

    def get_vector_batch(self, iterator, freezed=True):
        if not freezed:
            self.vocab['__DUMMY__'] = 0
            feature_values = list()
            feature_value_count = dict()
            for pred_word, arg_word, frame, sentence in iterator():
                feature_value = self.get_value(pred_word, arg_word, frame, sentence)
                feature_values.append(feature_value)
                feature_value_count[feature_value] = feature_value_count.get(feature_value, 0) +1
            feature_vectors = list()
            for feature_value in feature_values:
                if feature_value is not None and feature_value_count[feature_value] >= self.min_count:
                    feature_idx = self.vocab.setdefault(feature_value, len(self.vocab))
                    feature_vectors.append({feature_idx: 1})
                else:
                    feature_vectors.append({self.vocab['__DUMMY__']: 1})
        else:
            return super().get_vector_batch(iterator, freezed)
        return feature_vectors

    def get_vocab(self):
        return {self.vocab_name: self.vocab.copy()}

    def set_vocab(self, vocabs):
        if self.vocab_name in vocabs:
            self.vocab = vocabs[self.vocab_name]

    def __len__(self):
        return len(self.vocab)


class POSVocabFeature(EssentialVocabFeature, ElementMixin):

    vocab = dict()

    def __init__(self, name=None):
        self.vocab = POSVocabFeature.vocab
        self.element = 'gpos'
        super(POSVocabFeature, self).__init__(name)

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")


class LemmaVocabFeature(EssentialVocabFeature, ElementMixin):

    vocab = dict()

    def __init__(self, name=None):
        self.vocab = LemmaVocabFeature.vocab
        self.element = 'lemma'
        super(LemmaVocabFeature, self).__init__(name)

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")


class FormVocabFeature(EssentialVocabFeature, ElementMixin):

    vocab = dict()

    def __init__(self, name=None):
        self.vocab = FormVocabFeature.vocab
        self.element = 'form'
        super(FormVocabFeature, self).__init__(name)

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")


class DeprelVocabFeature(EssentialVocabFeature, ElementMixin):

    vocab = dict()

    def __init__(self, name=None):
        self.vocab = DeprelVocabFeature.vocab
        self.element = 'deprel'
        super(DeprelVocabFeature, self).__init__(name)

    def get_value(self, pred_word, arg_word, frame, sentence):
        raise NotImplementedError("Please Implement this method")


class PredSense(UniqueVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return frame.pred


class PredPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(pred_word)


class PredDeprel(DeprelVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(pred_word)


class ArgPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(arg_word)


class ArgLeftPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        left_word = sentence.get_leftmost_dep_word(arg_word)
        return self.get_element_value(left_word)


class ArgLeftForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        left_word = sentence.get_leftmost_dep_word(arg_word)
        return self.get_element_value(left_word)


class ArgRightPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        right_word = sentence.get_rightmost_dep_word(arg_word)
        return self.get_element_value(right_word)


class PredLemma(LemmaVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(pred_word)


class ArgLemma(LemmaVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(arg_word)


class ArgDeprel(DeprelVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(arg_word)


class PredForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(pred_word)


class ArgForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        return self.get_element_value(arg_word)


class ArgRightForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        right_word = sentence.get_rightmost_dep_word(arg_word)
        return self.get_element_value(right_word)


class Position(Feature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        distance = pred_word.word_id - arg_word.word_id
        if distance < 0:
            return 'left'
        if distance > 0:
            return 'right'
        return 'same'

    def get_idx(self, pred_word, arg_word, frame, sentence, freezed=False):
        indexes = dict(left=0, same=1, right=2)
        value_index = indexes[self.get_value(pred_word, arg_word, frame, sentence)]
        return value_index

    def __len__(self):
        return 3


class POSPath(UniqueVocabFeature):
    """PathFeature.java:
    Possible Bug? first word's pos is in string, last word's pos not. Either both or None in my opinion"""

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        pred_word_id = pred_word.word_id
        arg_word_id = arg_word.word_id
        if pred_word_id == arg_word_id:
            return ' '
        pos_path_str = 'NODEP'
        # 0 = UP; 1 = DOWN
        up = '1' if pred_word_id < arg_word_id else '0'
        for i in range(min(pred_word_id, arg_word_id)+1, max(pred_word_id, arg_word_id)):
            pos_path_str += sentence.get_word(i).get_element('gpos')
            pos_path_str += up
        #print(pos_path_str)
        return pos_path_str


class POSDepPath(UniqueVocabFeature):

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        pos_dep_path = ''
        dep_path, _ = sentence.get_dep_path(pred_word, arg_word)
        for word in dep_path:
            pos_dep_path += word.get_element('gpos')
        return pos_dep_path


class DeprelPath(UniqueVocabFeature):

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        deprel_dep_path = ''
        dep_path, _ = sentence.get_dep_path(pred_word, arg_word)
        for word in dep_path:
            deprel_dep_path += word.get_element('deprel')
        return deprel_dep_path


class PredChildFormSet(UniqueVocabFeature):

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        return ' '.join([w.get_element('form') for w in pred_word.direct_children])


class PredChildDepSet(UniqueVocabFeature):

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        return ' '.join([w.get_element('deprel') for w in pred_word.direct_children])


class PredChildPOSSet(UniqueVocabFeature):

    def __init__(self):
        super().__init__(min_count=3)

    def get_value(self, pred_word, arg_word, frame, sentence):
        return ' '.join([w.get_element('gpos') for w in pred_word.direct_children])


class PredParentForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        if pred_word.head == 0:
            return None
        parent_word = sentence.get_word(pred_word.head)
        return self.get_element_value(parent_word)


class PredParentPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        if pred_word.head == 0:
            return None
        parent_word = sentence.get_word(pred_word.head)
        return self.get_element_value(parent_word)


class ArgRightSiblingForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        right_sibling = sentence.get_right_sibling_word(arg_word)
        return self.get_element_value(right_sibling)


class ArgLeftSiblingForm(FormVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        left_sibling = sentence.get_left_sibling_word(arg_word)
        return self.get_element_value(left_sibling)


class ArgLeftSiblingPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        left_sibling = sentence.get_left_sibling_word(arg_word)
        return self.get_element_value(left_sibling)


class ArgRightSiblingPOS(POSVocabFeature):

    def __init__(self):
        super().__init__()

    def get_value(self, pred_word, arg_word, frame, sentence):
        right_sibling = sentence.get_right_sibling_word(arg_word)
        return self.get_element_value(right_sibling)


class LSTMFeature():

    def __init__(self):
        self.name = type(self).__name__
        self.pos_vocab = POSVocabFeature()
        self.form_vocab = FormVocabFeature()
        self.deprel_vocab = DeprelVocabFeature()

    def get_value(self, pred_word, arg_word, frame, sentence):
        path, common_idx = sentence.get_dep_path(pred_word, arg_word)
        value_path = list()
        for word in path[:common_idx]:
            value_path.append((word.gpos, word.form, (word.deprel, 1)))
        word = path[common_idx]
        value_path.append((word.gpos, word.form))
        for word in path[common_idx+1:]:
            value_path.append((word.gpos, word.form, (word.deprel, 0)))
        return value_path

    def _get_pos_vector(self, gpos):
        return {self.pos_vocab.get_idx_by_value(gpos): 1}

    def _get_form_vector(self, form):
        idx = self.form_vocab.get_idx_by_value(form)
        return {len(self.pos_vocab) + idx: 1}

    def _get_deprel_vector(self, deprel, direction):
        idx = self.deprel_vocab.get_idx_by_value(deprel)
        offset = len(self.pos_vocab) + len(self.form_vocab) + (len(self.deprel_vocab) * direction)
        return {offset + idx: 1}

    def get_vectors(self, pred_word, arg_word, frame, sentence):
        vectors = list()

        path, common_idx = sentence.get_dep_path(pred_word, arg_word)
        for word in path[:common_idx]:
            vectors.append(self._get_pos_vector(word.gpos))
            vectors.append(self._get_form_vector(word.form))
            vectors.append(self._get_deprel_vector(word.deprel, 1))
        word = path[common_idx]
        vectors.append(self._get_pos_vector(word.gpos))
        vectors.append(self._get_form_vector(word.form))
        for word in path[common_idx+1:]:
            vectors.append(self._get_pos_vector(word.gpos))
            vectors.append(self._get_form_vector(word.form))
            vectors.append(self._get_deprel_vector(word.deprel, 0))
        return vectors

    def get_vector_batch(self, iterator, freezed=True):
        feature_vectors = list()
        for pred_word, arg_word, frame, sentence in iterator():
            feature_vector = self.get_vectors(pred_word, arg_word, frame, sentence)
            feature_vectors.append(feature_vector)
        return feature_vectors

    def __len__(self):
        return len(self.pos_vocab) + len(self.form_vocab) + (2 * len(self.deprel_vocab))


essential_vocab_features = {'form': FormVocabFeature, 'lemma': LemmaVocabFeature, 'gpos': POSVocabFeature,
                            'deprel': DeprelVocabFeature}

available_features = {f.__name__: f for f in [
    PredSense, PredForm, PredLemma, PredPOS, PredDeprel,

    ArgForm, ArgLemma, ArgPOS, ArgDeprel, ArgLeftForm, ArgRightForm, ArgLeftPOS, ArgRightPOS,

    ArgRightSiblingForm, ArgLeftSiblingForm, ArgLeftSiblingPOS, ArgRightSiblingPOS,

    POSPath, POSDepPath, DeprelPath,

    PredChildFormSet, PredChildDepSet, PredChildPOSSet, PredParentForm, PredParentPOS,

    Position
    ]}