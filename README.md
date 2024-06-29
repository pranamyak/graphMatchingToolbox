# Graph Matching Toolbox

## Overview
The Graph Matching Toolbox provides implementations of various graph matching algorithms. It is designed to facilitate easy experimentation and evaluation of different models.

## Example Command
To run a model, you can use the following command:

```sh
python3 -m src.train --experiment_id simgnn --experiment_dir experiments/ --dataset_name 'aids' --seed 7762 --model_config_path configs/simgnn.yaml --dataset_size large --margin 0.5
