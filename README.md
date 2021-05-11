# 674-Final-Project

We distributed the tasks and implemented the SECOND Network as follows:

Pranjali:
builder/anchor_generator_builder.py: Line 8-26
builder/dataset_builder.py : 
11-30 Inline comments
37-43 Assertions to handle inconsistencies
builder/db_sampler_builder.py - Given the sampler configuration, a builder is created to sample the database
core/anchor_generator.py:
102-109 - Anchor generator function
core/box_coders.py
62-83 - Encode decode code optimization
core/geometry.py
5-37 - Added functions to see if 2 line segments intersect
data/kitti_dataset.py
58-67 - Bounding boxes visualization predictions and setting the min and max xy coordinates
104-120 - Inline comments
92-95 - _dict_select function
create_data.py: Script to create kitti dataset
script.py: Added functions to train the neural network and evaluating multiple threshold using model checkpoints 

Sowmya Vasuki:
/pytorch/models/middle.py:
195-203: Forward pass function
278-286: Forward pass function
458-456: Forward pass function
524-532: Forward pass function
613-621: Forward pass function
pytorch/Inference.py inline comments
data/kitti_common.py
25-44: Function to generate pairwise intersection areas between boxes
pytorch/models/resnet.py - Define 3x3 convolution with padding and 1x1 convolution with classes SparseBasicBlock and SparseBottleneck

Disha Singh:
Wrote builder/similary_calculator_builder.py : Lns 8-27
Re-Wrote builder/target_assigner_builder: Lns 18-37
Comments in core/target_ops.py : Lns 99 -133
Simplified similarity function in core/target_assigner.py  : 65-68
core/preprocess.py: removed irrelevant print statements
All_dataset.py : lines 128-143
Kitti_common.py wrote area function 12-21
Kitti_common.py wrote iou function 54-66
Kitti_common.py remove-low_height 323-331
pytorch/box_coder_builder.py
Added errors in input_reader_builder.py ln: 64-66
pytorch/losses_builder: added exceptions
pytorch/lr_scheduler_builder.py: added ‘rms_prop_optimizer’. Lns: 36-39
pytorch/optimizer_builder.py: added ‘rms_prop_optimizer’. Lns: 53-59





