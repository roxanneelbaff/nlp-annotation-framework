from textmining_utility import logger
from textmining_utility.annotator.orchestrator import PipeOrchestrator


class DummyDefaultPipeOrchestrator(PipeOrchestrator):
    pass




def customized_pipe_func(input):
    logger.debug("checking Dummy customized pipe")
    return input