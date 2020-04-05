
class AgentBase:
    """ Basic interface of agents
    """
    def initialize(self, observation_space, action_space):
        """ Building the actual model for computations given two spaces
        """
        raise NotImplementedError

    def to(self, device):
        """ A torch like method to move neural networks into CUDA
        """
        pass

    def state_dict(self):
        """ return a state dicts that stores model parameters, just like torch.Tensor.state_dict()
        """
        return dict()

    def train_mode(self):
        """ Just like nn.Module.train()
        """
        pass

    def sample_mode(self):
        """ Jut like nn.Module.eval()
        """
        pass

    def load_state_dict(self, state_dict):
        """ given a state_dict with the same format this instance outputs, load snapshot to this
        instance.
        """
        pass

    def step(self, observation):
        """ Given the observation, the agent outputs the action. For recurrent agent, it stores
        hidden states itself, and abandom them via reset.
        NOTE: The input and output should all be batch-wise
        @ Args
            observation: torch.Tensor with appropriate shape
        @ Returns
            action: torch.Tensor with appropriate shape
        """
        raise NotImplementedError

    def reset(self, batch_size= 1):
        """ NOTE: reset might be called many times, but initialize will be called only once.
        And do NOT reset learnt parameters in here, setup them in initialize()
        """
        self.batch_size = batch_size
