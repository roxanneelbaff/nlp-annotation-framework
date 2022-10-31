import pandas
import pandas as pd
from pandas import DataFrame

from textmining_utility.annotator.component.annotator import PipeOrchestrator


class EmotionOrchestrator(PipeOrchestrator):
    _EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


    def save_annotations(self, annotated_texts: list, folder_path="") -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            for emotion_score in doc._.emotion_hartmann_scores:
                row[f"emotion_hartmann_{emotion_score['label']}"] = emotion_score['score']

            row[f"emotion_hartmann_label"] = doc._.emotion_hartmann_dominant
            out_arr.append(row)
        out_df: DataFrame = pd.DataFrame(out_arr)

        return out_df









