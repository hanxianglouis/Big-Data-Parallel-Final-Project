# Claim
This repository does not aim for any paper publication. This is just used to share our final project of a course. All the models and algorithms are not designed by us, while all of the code are written by ourselves. Everyone is welcomed to use these code.
# Big Data Parallel Computing Final Project (2025 Fall Renmin Univ. of China)
## About
In this project, we use a simplified Decision Focused Causal Learning(DFCL) model to do some causal inferences based on the dataset [CRITEO-UPLIFT v2](https://huggingface.co/datasets/criteo/criteo-uplift). Because the original implementation of this model are written based on Tensorflow, which is difficult to achieve parallel computing, we rewrite the code based on Torch. Unfortunately, limited by our own capabilities, we abandoned some modules in the DFCL model such as the Hashing layer and Embedding layer.
## How to Run
If you want to run the code, you are supposed to download the raw dataset, and divide it into two parts `criteo_train.csv` and `criteo_val.csv`. And you should modify the path of these two files, which are at row 95 and 96 in `run.py`. We highly recommend you to put all your dataset in a folder named "data", although it will be created automatically the first time you run the code.

The Python file `run_single.py` can be run at Windows, MacOS and Linux. And if your device is equipped with CUDA environment or MPS (Apple Silicon), the code will use GPU automatically to accelerate. The total number of paramaters is 213,892, therefore using GPU is suggested.

However, if you want to run a parallel version, you should run `run_parallel.py` which can only run in a cuda environment. To run this script, enter
``` bash
torchrun --standalone --nproc_per_node=k run_parallel.py
```
in the terminal, and substitude the $k$ by the number of GPUs you have.

To test the time consumption, we set `epoches=1`. To train the model, you can set it to a large number, and you can also set the paramater `patience`, which means that if the model does not imporve for $patience$ epochs, the train will end. Moreover, the best model (selected based on validate loss) will be stored in the folder `model`. But in some situations, such as in the matpool platform, the model cannot be stored due to the limited space of disk. In this case, the metrics you get will be from the final trained model that may not be the best one.

## Findings
We removed the `log1p` function in data preprocessing because it may cause `Nan` value. In MPS device, it will be ignored, but not in CUDA environment. So we just applied normalization on the original data.