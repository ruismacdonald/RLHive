import torch
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(
    torch.nn.Module,
    {
        "SimpleMLP": SimpleMLP,
        "SimpleConvModel": SimpleConvModel,
        "NatureAtariDQNModel": NatureAtariDQNModel,
    },
)
