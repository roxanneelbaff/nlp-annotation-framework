
from spacy_arguing_lexicon import ArguingLexiconParser

from spacy.language import Language
from spacy.tokens import Token, Span, Doc

from transformers.utils import logging


@Language.factory("mpqaW_component")
class ToxicityFactory:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        logging.disable_progress_bar()


        mpqa_labels = [    
            'assessments',
            'doubt',
            'authority',
            'emphasis',
            'necessity',
            'causation',
            'generalization',
            'structure',
            'conditionals',
            'inconsistency',
            'possibility',
            'wants',
            'contrast',
            'priority',
            'difficulty',
            'inyourshoes',
            'rhetoricalquestion',
           
        ]

        mpqa_additional_labels = [
             'argumentative', 
            'mpqa_token_ratio',
            'count_mpqa_args'
        ]
        mpqa_labels = [f"mpqa_{x}" for x in mpqa_labels+mpqa_additional_labels]
        for label in mpqa_labels:
            if not Doc.has_extension(label):
                Doc.set_extension(label, default=0)




    def __call__(self, doc):
        arguments = list(doc._.arguments.get_argument_spans_and_matches())
        doc._.argumentative = len(arguments)
        total_arg_words = 0

        for arg in arguments:
            arg_span = arg[0]
            label = arg_span.label_

            col = "{}_{}".format("mpqa", label)
            doc._.set(col, doc._.get(col)+1)

            total_arg_words += arg_span.__len__()

        doc._.mpqa_token_ratio =round(float(total_arg_words) / float(doc.__len__()), 3)
        doc._.count_mpqa_args = len(arguments)

        return doc


@Language.factory("mpqa_parser")
def mpqa_parser(nlp, name):
    return ArguingLexiconParser(lang=nlp.lang)