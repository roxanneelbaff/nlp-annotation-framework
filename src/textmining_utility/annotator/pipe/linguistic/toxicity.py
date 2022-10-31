import pandas as pd
from simpletransformers.ner import NERModel
from spacy.language import Language
from spacy.tokens import Token, Span, Doc
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# REQUIRES sentencizer
#https://huggingface.co/SkolkovoInstitute/roberta_toxicity_classifier?text=I+like+you.+I+love+you

@Language.factory("toxicity_component")
class ToxicityFactory:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp

        model_path = 'SkolkovoInstitute/roberta_toxicity_classifier'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)

        self.transformer_nlp = pipeline('text-classification',
                                        model=model,
                                        tokenizer=tokenizer)

        if not Span.has_extension("toxicity_neutral"):
            Span.set_extension("toxicity_neutral", default=None)

        if not Span.has_extension("toxicity_toxic"):
            Span.set_extension("toxicity_toxic", default=None)

        if not Span.has_extension("toxicity_dominant"):
            Span.set_extension("toxicity_dominant", default=None)

        if not Doc.has_extension("toxicity"):
            Doc.set_extension("toxicity", default=None)
    def __call__(self, doc):
        all_scores = {}
        all_scores["neutral"] = []
        all_scores["toxic"] = []

        sentence_lbls = []

        for sentence in doc.sents:
            txt = sentence.text
            sent_toxicity_result = transformer_nlp(txt)

            dominant_lbl = None
            score_holder = 0.0

            for sent_toxicity_score in sent_toxicity_result:
                label = sent_toxicity_score['label']
                score = sent_toxicity_score['score']
                if dominant_lbl is None or score_holder < score:
                    dominant_lbl = label
                    score_holder = score

                all_scores[label].append(score)

                sentence._.set(f"toxicity_{label}", score)
            sentence._.set(f"toxicity_dominant", dominant_lbl)
            sentence_lbls.append(dominant_lbl)


        doc_mean_df =  pd.DataFrame(
            [{
                "label":x,
                "raw_mean":  mean(y) }
            for x, y in all_scores.items() ]).set_index("label")


        doc_scores_df = pd.DataFrame([
            {
                "label":x,
                "count": sentence_lbls.count(x),
                "ratio":round(sentence_lbls.count(x)/len(sentence_lbls),3)
            }
            for x in set(classifications)
        ]).set_index("label")

        predictions = doc_scores_df.merge(doc_mean_df,
                                         how='inner',
                                         left_index=True,
                                         right_index=True).reset_index().to_dict('records')


        doc._.set("toxicity", predictions)

        return doc