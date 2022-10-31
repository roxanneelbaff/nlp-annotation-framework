_dict = {}

########## FIELDS ##########
orchestrator_class = "orchestrator_class"
pipe_id_or_func = "pipe_id_or_func"
pipe_path = "pipe_path"  # leave empty if the pipe is a spacy native pipe
level = "level"  # node or set
data_type = "data_type"  # for now we support text - later url, image
annotated_column = "annotated_column"
input_id = "input_id"



_dict["NamedEntityPipeOrchestrator"] = {
    orchestrator_class : "orchestrator_class",
    pipe_id_or_func : "ner", # id in case of space, function name otherwise
    pipe_path : "",  # leave empty if the pipe is a spacy native pipe, otherwise provide the path of where the pipe_id_or_func exists
    level : "singelton",  # node or set
    data_type : "text",  # for now we support text - later url, image
    annotated_column : "text",
    input_id : "input_id"

}
