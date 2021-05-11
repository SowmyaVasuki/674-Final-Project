## 674-Final-Project

We distributed the tasks and implemented the SECOND Network as follows:

### Pranjali Ajay Parse:
- 8-26 - builder/anchor_generator_builder.py
#### builder/dataset_builder.py : 
- 11-30 - Inline comments
- 37-43 - Assertions to handle inconsistencies
- builder/db_sampler_builder.py - Given the sampler configuration, a builder is created to sample the database3
#### core/anchor_generator.py:
- 102-109 - Anchor generator function
#### core/box_coders.py:
- 62-83 - Encode decode code optimization
#### core/geometry.py:
- 5-37 - Added functions to see if 2 line segments intersect
#### data/kitti_dataset.py:
- 58-67 - Bounding boxes visualization predictions and setting the min and max xy coordinates
- 104-120 - Inline comments
- 92-95 - _dict_select function

- create_data.py: Script to create kitti dataset
- script.py: Added functions to train the neural network and evaluating multiple threshold using model checkpoints 

### Sowmya Vasuki:
#### pytorch/models/middle.py:
- 195-203: Forward pass function
- 278-286: Forward pass function
- 458-456: Forward pass function
- 524-532: Forward pass function
- 613-621: Forward pass function
- pytorch/Inference.py inline comments

#### data/kitti_common.py
- 25-44: Function to generate pairwise intersection areas between boxes
- pytorch/models/resnet.py - Define 3x3 convolution with padding and 1x1 convolution with classes SparseBasicBlock and SparseBottleneck

### Disha Singh:
- 8-27 - Wrote builder/similary_calculator_builder.py
- 18-37 - Re-Wrote builder/target_assigner_builder
- 99 -133 - Comments in core/target_ops.py
- 65-68 - Simplified similarity function in core/target_assigner.py
- core/preprocess.py: removed irrelevant print statements
- All_dataset.py : lines 128-143
- pytorch/box_coder_builder.py
- 64-64 - Added errors in input_reader_builder.py
- pytorch/losses_builder: added exceptions
- 36-39 - pytorch/lr_scheduler_builder.py: added ‘rms_prop_optimizer’
- 53-59 - pytorch/optimizer_builder.py: added ‘rms_prop_optimizer’

#### Kitti_common.py 
- 12-21 - wrote area function
- 54-66 - wrote iou function
- 323-331 - remove-low_height




