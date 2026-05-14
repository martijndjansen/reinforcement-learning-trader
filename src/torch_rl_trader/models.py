import torch.nn as nn
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torchrl.modules import MLP, NormalParamExtractor, TanhNormal, ValueOperator


def build_mlp_sac_modules(
    obs_dim: int, act_dim: int, hidden_sizes: list[int]
) -> tuple[ProbabilisticTensorDictSequential, ValueOperator]:

    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=hidden_sizes,
            norm_class=nn.LayerNorm,
            norm_kwargs={"normalized_shape": hidden_sizes[-1]},
        ),
        NormalParamExtractor(),
    )

    q_net = MLP(
        in_features=obs_dim + act_dim,
        out_features=1,
        num_cells=hidden_sizes,
        norm_class=nn.LayerNorm,
        norm_kwargs={"normalized_shape": hidden_sizes[-1]},
    )

    policy = ProbabilisticTensorDictSequential(
        [
            TensorDictModule(
                actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                distribution_kwargs={"low": -10.0, "high": 10.0},
                return_log_prob=True,
            ),
        ]
    )

    q_module = ValueOperator(
        module=q_net,
        in_keys=["observation", "action"],
    )

    return policy, q_module
