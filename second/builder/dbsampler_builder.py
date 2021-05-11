import pickle
import second.core.preprocess as prep
from second.builder import preprocess_builder
from second.core.sample_ops import DataBaseSamplerV2
from second.core.preprocess import DataBasePreprocessor

"""
Database sampler builder given the sampler configuartion
"""
def build(sampler_config):
    cfg = sampler_config
    rate = cfg.rate # DB Sample builder rate
    groups = list(cfg.sample_groups) # Building sample groups
    info_path = cfg.database_info_path # Info path
    prepors = [preprocess_builder.build_db_preprocess(cf_g) for cf_g in cfg.database_prep_steps]
    db_prepor = DataBasePreprocessor(prepors)
    
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [dict(group.name_to_max_num) for group in groups]
    
    with open(info_path, 'rb') as f: db_infos = pickle.load(f)
    grot_range_list = list(grot_range) # List of grot range

    if len(grot_range_list) == 0: grot_range_list = None

    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range_list)
    return sampler
