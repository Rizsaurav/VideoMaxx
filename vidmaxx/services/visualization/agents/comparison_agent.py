from vidmaxx.services.visualization.base_agent import BaseVizAgent
from vidmaxx.services.visualization.config_models import ComparisonConfig
from vidmaxx.services.visualization.prompts.comparison import DEBUG, EXTRACT, TEMPLATE


class ComparisonAgent(BaseVizAgent):
    viz_type = "comparison"
    template = TEMPLATE
    extract_prompt = EXTRACT
    debug_prompt = DEBUG
    config_model = ComparisonConfig
