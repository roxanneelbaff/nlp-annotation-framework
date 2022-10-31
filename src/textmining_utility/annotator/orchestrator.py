import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd
import textmining_utility.annotator.registered_pipes as registered
from textmining_utility import logger

@dataclasses.dataclass
class PipeOrchestrator(ABC):
    orchestrator: str = None
    registered_pipes: Dict = None
    load_default: bool = True

    _ANNOTATION_TYPE : ClassVar = ["text", "url", "image"]
    _ANNOTATION_LEVEL: ClassVar = ["singleton", "collection"]

    def _load_registered_pipes(self):
        if self.load_default:
            _copy = registered._dict.copy()
            if self.registered_pipes is not None:
                _copy.update(self.registered_pipes)
            self.registered_pipes = _copy
        return self.registered_pipes


    def __post_init__(self):
        self._load_registered_pipes()
        self.name = self.__class__.__name__ if orchestrator is None else orchestrator
        self.pipes_config = registered_pipes[self.name]
        self.annotation_level = self.pipes_config["level"]
        self.annotation_type = self.pipes_config["data_type"]
        self.annotated_column = self.pipes_config[ "annotated_column"]
        self.pipe_id_or_func = self.pipes_config["pipe_id_or_func"] if "pipe_id_or_func" in self.pipes_config.keys() else None
        self.pipe_path = self.pipes_config["pipe_path"] if "pipe_path" in self.pipes_config.keys() else None
        self.input_id = self.pipes_config["input_id"]

        self.load_pipe_component()

        is_valid, error_dic = self.validate()
        if is_valid:
            logger.info("orchestrator was initialized successfully")
        else:
            logger.error(f"orchestrator was initialized with the following errors {error_dic}")

    def load_pipe_component(self):
        module = None
        if self.pipe_path is not None and len(self.pipe_path) > 0:
            logger.debug("loading {}".format(self.pipe_path))
            module = utils.load_module(self.pipe_path)
        return module

    def get_pipe_func(self):
        module = self.load_pipe_component()
        pipe_func = None
        if module is not None  and self.pipe_id_or_func is not None:
            pipe_func = getattr(module, self.pipe_id_or_func)
        return pipe_func


    def validate(self):
        error_dict = {}
        if self.annotation_type not in PipeOrchestrator._ANNOTATION_TYPE:
            error_dict["annotation_type"] = "The annotation type should have one of the following values: {} but has the value {}".\
                format(str(PipeOrchestrator.ANNOTATION_TYPE), self.annotation_type)
            logger.error(error_dict["annotation_type"])

        if self.annotation_level not in PipeOrchestrator._ANNOTATION_LEVEL:
            error_dict["annotation_level"] = "The annotation level should have one of the following values: {} " \
                                         "but has the value {}".\
                                          format(str(PipeOrchestrator.ANNOTATION_LEVEL), self.annotation_level)
            logger.error(error_dict["annotation_level"])

        is_valid = not bool(error_dict)
        return is_valid, error_dict


    @abstractmethod
    def save_annotations(self, annotated_text: pd.DataFrame):
        pass

    @abstractmethod
    def run(self, input: pd.DataFrame):
        pass


