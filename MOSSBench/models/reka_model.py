import reka

from .model_base import BlackBoxModelBase

class RekaModel(BlackBoxModelBase):
    def __init__(self, model_name: str, api_keys: str, generation_config=None):

        reka.API_KEY = api_keys
        self.model_name = model_name
        self.seed = 42

    def generate(self, message, image, clear_old_history=True, **kwargs):
        num_attempts = 0
        while num_attempts < 5:
            num_attempts += 1
            try:
                response = reka.chat(message,
                                     model_name=self.model_name,
                                     random_seed=self.seed,
                                     temperature=1,
                                     runtime_top_p=0.9,
                                     media_filename=image)
                return response['text']
            except Exception as e:
                print(e)

