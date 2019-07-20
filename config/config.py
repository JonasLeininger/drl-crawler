import yaml
import torch
from unityagents import UnityEnvironment

class Config:

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_agents = None
        self.config = self.load_config_file()
        self.env = UnityEnvironment(file_name=self.config['CrawlerSingle'])
        self.init_env()

    def load_config_file(self, config_file: str = 'config/config.yaml'):
        with open(config_file, 'r') as stream:
            try:
                return yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as ex:
                print(ex)

    def init_env(self):
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.states = self.env_info.vector_observations
        self.state_dim = self.states.shape[1]
        self.action_dim = self.brain.vector_action_space_size
    
    def close_env(self):
        self.env.close()
