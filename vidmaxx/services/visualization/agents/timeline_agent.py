from vidmaxx.services.visualization.base_agent import BaseVizAgent
from vidmaxx.services.visualization.config_models import TimelineConfig
from vidmaxx.services.visualization.prompts.timeline import DEBUG, EXTRACT, TEMPLATE


class TimelineAgent(BaseVizAgent):
    viz_type = "timeline"
    template = TEMPLATE
    extract_prompt = EXTRACT
    debug_prompt = DEBUG
    config_model = TimelineConfig
