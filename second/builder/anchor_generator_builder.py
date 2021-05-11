import numpy as np

from second.protos import box_coder_pb2
from second.core.anchor_generator import (AnchorGeneratorStride,
                                          AnchorGeneratorRange)


anchor_stride = 'anchor_generator_stride' # ag type generator stride
anchor_range = 'anchor_generator_range' # ag type generator range
no_anchor = 'no_anchor' # ag type no anchor

def build(class_cfg):
    """Create an optimizer according to the config
  @args: optimizer_config
  Returns: optimizer, list of variables for summary.
  """
    ag_type = class_cfg.WhichOneof('anchor_generator')
    ag = None
    if ag_type == anchor_stride:
      config = class_cfg.anchor_generator_stride
      ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))

    elif ag_type == anchor_range:
      config = class_cfg.anchor_generator_range
      ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))
    elif ag_type == no_anchor:
      pass
    return ag
