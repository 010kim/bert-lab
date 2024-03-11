import collections
import six
import re
import unicodedata

def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name"""

    if not init_checkpoint:
        return
    
    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return
    
    model_name = m.group(1)
    lower_models = ["uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12", "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"]
    cased_models = ["cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16", "multi_cased_L-12_H-768_A-12"]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError("do_lower_case and model do not match")


def convert_to_unicode(text):
    """Convert text to Unicode, assuming utf-8 input"""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
    

class FullTokenizer():
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        #self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        
        def tokenize(self, text):
            raise NotImplementedError
        
        def convert_tokens_to_ids(self, tokens):
            raise NotImplementedError
        
        def convert_ids_to_tokens(sef, ids):
            raise NotImplementedError


class BasicTokenizer():
    """punctuation splitting, lower casing, etc"""
    
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        
    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(self, text)
        
    def _clean_text(self, text):
        """invalid character removal, whitespace cleanup"""
        output = []
        for char in text:
            cp = ord(char)
            #continue workin 240311


class WordpieceTokenizer():
    pass

