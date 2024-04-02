from abc import ABC, abstractmethod


class AgentInterface(ABC):
    """
    L'interface requise par tous les agents.
    """

    @abstractmethod
    def select_action(self, state):
        """
        Select an action given the current policy and a state
        """
        pass

    def select_greedy_action(self, state):
        return self.select_action(state)
