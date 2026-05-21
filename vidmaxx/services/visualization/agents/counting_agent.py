from vidmaxx.services.visualization.base_agent import BaseVizAgent
from vidmaxx.services.visualization.config_models import CountingConfig
from vidmaxx.services.visualization.prompts.counting import DEBUG, EXTRACT, TEMPLATE


class CountingAgent(BaseVizAgent):
    viz_type = "counting_number"
    template = TEMPLATE
    extract_prompt = EXTRACT
    debug_prompt = DEBUG
    config_model = CountingConfig
