import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.distributions import Dirichlet


class DirichletDistribution(CategoricalDistribution):
    """Dirichlet distribution"""

    distribution: Dirichlet  # type: ignore

    def __init__(self, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__(action_dim=action_dim)

        self.hidden_dim = hidden_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Dirichlet distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        return nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softplus(),
        )

    def proba_distribution(self, action_logits: th.Tensor) -> Distribution:  # type: ignore
        self.distribution = Dirichlet(concentration=action_logits)  # type: ignore
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mean()
        return self.sample()

    def mean(self) -> th.Tensor:
        """This is the mean"""
        alpha = self.distribution.concentration
        return alpha / alpha.sum(dim=-1, keepdim=True)

    def actions_from_params(
        self, action_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class DirichletPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        if self.action_space.shape:
            self.action_dist = DirichletDistribution(self.action_space.shape[0])
        else:
            raise ValueError()

        super()._build(lr_schedule)
