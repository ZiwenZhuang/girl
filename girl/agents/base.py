
class AgentBase:
    """ Basic interface of agents
    """
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
