import pandas
import pandas as pd
from pandas import DataFrame

from textmining_utility.annotator.component.annotator import PipeOrchestrator


class HedgeOrchestrator(PipeOrchestrator):


    def save_annotations(self, annotated_texts: list, folder_path="") -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            if doc._.hedge is not None and len(doc._.hedge) > 0:
                for hedge_dict in doc._.hedge:
                    row[f"hedge_{hedge_dict['label']}_count"] = hedge_dict['size']
                    row[f"hedge_{hedge_dict['label']}_ratio"] = hedge_dict['ratio']
                    row[f"hedge_{hedge_dict['label']}_words"] = hedge_dict['words']

                res_df = pd.DataFrame(doc._.hedge  )

                row[f"hedge_dominant"] =res_df.iloc[res_df["size"].argmax()]["label"]
                out_arr.append(row)
        out_df: DataFrame = pd.DataFrame(out_arr)

        return out_df









