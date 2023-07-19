from .layers import MLP, GluMLP, GatedMLP

MLP_LAYERS = {
    "mlp": MLP,
    'glu_mlp': GluMLP,
    "gated_mlp": GatedMLP
}