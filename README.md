# MFM-point
We include simplified code.

## How to Start
Go to ``metrics/PyTorchEMD`` and install setup by the following command:
```
conda setup.py install
```
Then, evaluation on CUDA code becomes available.


**Preprocessing**
Example:
```
python preprocessing.py --category airplane --npoints 2048 --downsample_ratio 4 --num_per_data 5
```

## Training 

Example:

- High res
```
python train_separate.py --exp_dir high_res --stage 0 --stage_start_time 0.6 --grad_clip 0.01
```

- Low res
```
python train_separate.py --exp_dir low_res --stage 1 --grad_clip 0.01
```

## Evaluation
Example:

```
python evaluate_separate.py --exp_names low_res high_res
```
