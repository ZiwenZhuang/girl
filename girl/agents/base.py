
class AgentBase:
    """ Basic interface of agents
    """
    def state_dict(self):
        """ return a state dicts that stores model parameters, just like torch.Tensor.state_dict()
        """
        return dict()

    def load_state_dict(self, state_dict):
        """ given a state_dict with the same format this instance outputs, load snapshot to this
        instance.
        """
        pass

    def step(self, observation):
        """ Given the observation, the agent outputs the action. For recurrent agent, it stores
        hidden states itself, and abandom them via reset.
        @ Args
            observation: torch.Tensor with appropriate shape
        @ Returns
            action: torch.Tensor with appropriate shape
        """
        raise NotImplementedError

    def reset(self):
        pass
