import pandas
import pandas as pd
from pandas import DataFrame

from textmining_utility.annotator.component.annotator import PipeOrchestrator


class EmpathPipeOrchestrator(PipeOrchestrator):

    def save_annotations(self, annotated_texts: list, folder_path="") -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            for category, count in doc._.empath_count.items():
                row[f"{category}_count"] = doc._.empath_count[category]
                row[f"{category}_ratio"] = doc._.empath_ratio[category]
                row[f"{category}_position"] = doc._.empath_positions[category]
                row[f"{category}_word"] = doc._.empath_words[category]
            out_arr.append(row)
        out_df: DataFrame = pd.DataFrame(out_arr)

        return out_df









