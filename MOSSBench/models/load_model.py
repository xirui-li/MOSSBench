from .anthropic_model import AnthropicModel
from .google_model import GoogleModel
from .huggingface_model import HuggingfaceModel, from_pretrained
from .openai_model import OpenaiModel
from .reka_model import RekaModel


def load_model(args):
    """
    Return the VLM model specified in args
    :param args: Specification
    :return: VLM model
    """
    if "gpt" in args.model_name:
        model = OpenaiModel(model_name=args.model_name,
                            api_keys=args.openai_api_key)
    elif "gemini" in args.model_name:
        model = GoogleModel(model_name=args.model_name,
                            api_key=args.google_key, safety_level=args.google_safety_level if args.google_safety_level else None)
    elif "claude" in args.model_name:
        model = AnthropicModel(model_name=args.model_name,
                               api_keys=args.anthropic_key)
    elif "grok" in args.model_name:
        raise NotImplementedError("Model Not implemented!")
    elif "reka" in args.model_name:
        model = RekaModel(model_name=args.model_name,
                          api_keys=args.reka_key)
    else:  # Open-source model
        model = from_pretrained(model_name_or_path=args.model_weight_path if args.model_weight_path else args.model_name,
                                model_name=args.model_name)

    return model