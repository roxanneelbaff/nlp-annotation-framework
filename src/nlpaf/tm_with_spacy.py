import spacy
from spacy_arguing_lexicon import ArguingLexiconParser
from empath import Empath
from spacy.language import Language
from spacy.tokens import Token, Span, Doc


class TM:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.add_pipe("merge_noun_chunk")
        self.nlp.add_pipe("descriptor")
        self.nlp.add_pipe("mpqa_parser")
        self.nlp.add_pipe("mpqa_counter")
        self.nlp.add_pipe("empath_component")


#### HELPERS ####
def set_extension_(ext, default=None):
    if not Doc.has_extension(ext):
        Doc.set_extension(ext, default=default)


def count_tokens_w_attr(doc, attr):
    return {doc.vocab[k].text: v for k, v in doc.count_by(attr).items()}


def get_content_pos(pos_tuple=("verb", "noun", "adjective", "adverb")):
    pos_labels = []
    for label in spacy.load("en_core_web_sm").get_pipe("tagger").labels:
        if spacy.explain(label).lower().startswith(pos_tuple):
            pos_labels.append(label)
    return pos_labels


### END OF Helpers ####


### Pipes ###
@Language.component("empath_component")
def empath_component_function(doc):
    lexicon = Empath()
    empath_dic = lexicon.analyze(doc.text, normalize=False)
    for k, v in empath_dic.items():
        col = "{}_{}".format("empath", k)
        set_extension_(col, default=None)
        doc._.set(col, v)
    return doc


@Language.component("mpqa_counter")
def mpqa_counter_function(doc):
    arguments = list(doc._.arguments.get_argument_spans_and_matches())

    total_arg_words = 0

    for arg in arguments:
        arg_span = arg[0]
        label = arg_span.label_

        col = "{}_{}".format("mpqa", label)
        set_extension_(col, default=0)

        doc._.set(col, doc._.get(col) + 1)

        total_arg_words += arg_span.__len__()

    set_extension_("mpqa_token_ratio", default=0.0)
    set_extension_("count_mpqa_args", default=0)

    doc._.mpqa_token_ratio = round(float(total_arg_words) / float(doc.__len__()), 3)
    doc._.count_mpqa_args = len(arguments)

    return doc


@Language.component("descriptor")
def descriptor_function(doc):
    CONTENT_POS = get_content_pos()
    if not Doc.has_extension("num_token"):
        Doc.set_extension("num_token", default=0)
    if not Doc.has_extension("num_content_tokens"):
        Doc.set_extension("num_content_tokens", default=0)

    doc._.num_token = doc.__len__()
    doc._.num_content_tokens = len(
        [token for token in doc if token.tag_ in CONTENT_POS]
    )

    pos_count_dict = count_tokens_w_attr(doc, spacy.attrs.POS)

    for pos_tag, count in pos_count_dict.items():
        pos_tag_name = "{}_{}".format("pos_num_", pos_tag)
        if not Doc.has_extension(pos_tag_name):
            Doc.set_extension(pos_tag_name, default=0)
        doc._.set(pos_tag_name, count)

    return doc


@Language.factory("mpqa_parser")
def mpqa_parser(nlp, name):
    return ArguingLexiconParser(lang=nlp.lang)
