from vidmaxx.services.visualization.base_agent import BaseVizAgent
from vidmaxx.services.visualization.config_models import FlowConfig
from vidmaxx.services.visualization.prompts.flow import DEBUG, EXTRACT, TEMPLATE


class FlowAgent(BaseVizAgent):
    viz_type = "flow"
    template = TEMPLATE
    extract_prompt = EXTRACT
    debug_prompt = DEBUG
    config_model = FlowConfig
