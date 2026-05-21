from vidmaxx.services.visualization.base_agent import BaseVizAgent
from vidmaxx.services.visualization.config_models import ShrinkConfig
from vidmaxx.services.visualization.prompts.shrink import DEBUG, EXTRACT, TEMPLATE


class ShrinkAgent(BaseVizAgent):
    viz_type = "shrink"
    template = TEMPLATE
    extract_prompt = EXTRACT
    debug_prompt = DEBUG
    config_model = ShrinkConfig
