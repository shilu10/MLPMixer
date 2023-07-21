from .blocks import MixerBlock, SpatialGatingBlock, ResBlock


BLOCK_LAYERS = {
    "mixer_block": MixerBlock,
    "spatial_gating_block": SpatialGatingBlock,
    "res_block": ResBlock
 }