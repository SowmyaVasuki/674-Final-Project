import numpy as np

from second.core import region_similarity
from second.protos import similarity_pb2


def build(config):
    """Create optimizer according to config.
    @args: optimizer_config
    Returns:
        An optimizer and a list of variables for summary.
    """
    type_similar = config.WhichOneof('region_similarity')
    type_similar_types = ['rotate_iou_similarity', 'nearest_iou_similarity', 'distance_similarity']
    value = None
    if type_similar == type_similar_types[0]:
        value = region_similarity.RotateIouSimilarity()
    elif type_similar == type_similar_types[1]:
        value = region_similarity.NearestIouSimilarity()
    elif type_similar == type_similar_types[2]:
        cfg = config.distance_similarity
        value = region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha)
    
    return value
    

