import numpy as np

from spconv.utils import VoxelGeneratorV2
from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary according to the InputReader config.

    @args: input_reader_config
    Returns: A tensor dict based on the input_reader_config.
    """
    voxel_generator = VoxelGeneratorV2(
        point_cloud_range=list(voxel_config.point_cloud_range),
        voxel_size=list(voxel_config.voxel_size),
        max_voxels=20000,
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        block_factor=voxel_config.block_factor,
        block_size=voxel_config.block_size,
        block_filtering=voxel_config.block_filtering,
        full_mean=voxel_config.full_empty_part_with_mean,
        height_threshold=voxel_config.height_threshold)
    return voxel_generator
