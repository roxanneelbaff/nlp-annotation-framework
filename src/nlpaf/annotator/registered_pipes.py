import dataclasses

_dict = {}

########## FIELDS ##########
@dataclasses.dataclass
class PipeConfigKeys:
    _orchestrator_class = "orchestrator_class"
    _pipe_id_or_func = "pipe_id_or_func"
    _pipe_path = "pipe_path"  # leave empty if the pipe is a spacy native pipe
    _level = "level"  # node or set
    _data_type = "data_type"  # for now we support text - later url, image
    _annotated_column = "annotated_column"
    _input_id = "input_id"



_dict["NamedEntityPipeOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "orchestrator_class",
    PipeConfigKeys._pipe_id_or_func : "ner", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}

_dict["EmotionPipeOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "nlpaf.annotator.instance.emotion_orchestrator.EmotionPipeOrchestrator",
    PipeConfigKeys._pipe_id_or_func : "emotion_hartmann_component", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "nlpaf.annotator.pipe.linguistic.emotion",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}

_dict["HedgePipeOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "nlpaf.annotator.instance.hedge_orchestrator.HedgeOrchestrator",
    PipeConfigKeys._pipe_id_or_func : "hedge_component", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "nlpaf.annotator.pipe.linguistic.hedge",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}

_dict["ToxicityOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "nlpaf.annotator.instance.toxicity_orchestrator.ToxicityOrchestrator",
    PipeConfigKeys._pipe_id_or_func : "toxicity_component", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "nlpaf.annotator.pipe.linguistic.toxicity",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}

_dict["MpqaPipeOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "nlpaf.annotator.instance.mpqa_orchestrator.MpqaPipeOrchestrator",
    PipeConfigKeys._pipe_id_or_func : "mpqa_arg_component", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "nlpaf.annotator.pipe.linguistic.mpqa",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}

_dict["EmpathPipeOrchestrator"] = {
    PipeConfigKeys._orchestrator_class : "nlpaf.annotator.instance.empath_orchestrator.EmpathPipeOrchestrator",
    PipeConfigKeys._pipe_id_or_func : "empath_component", # id in case of space, function name otherwise
    PipeConfigKeys._pipe_path : "nlpaf.annotator.pipe.linguistic.empath",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    PipeConfigKeys._level : "singleton",  # node or set
    PipeConfigKeys._data_type : "text",  # for now we support text - later url, image
    PipeConfigKeys._annotated_column : "text",
    PipeConfigKeys._input_id : "input_id"

}