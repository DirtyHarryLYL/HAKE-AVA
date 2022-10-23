## Getting Started with Activity2Vec

This document provides a introduction of using and finetuning the ST-Activity2Vec model. Please make sure you have properly installed the ST-Activity2Vec following the instruction in INSTALL.md and you have prepared the dataset following DATASET.md with the correct format.

### Evaluate

Load the pretrained ST-Activity2Vec model and evaluate the PaSta & action detection performance:

```
# Set TRAIN.VAL_ONLY to True to only evaluate the given checkpoint.
# TRAIN.PER_GPU_BS is the batch size per gpu.
python -u tools/run_net.py --cfg configs/HAKE/verb_base.yaml \
                            TRAIN.CHECKPOINT_FILE_PATH /your/checkpoint/path \
                            NUM_GPUS 8 \
                            TRAIN.PER_GPU_BS 1 \
                            TRAIN.ENABLE True \
                            TRAIN.VAL_ONLY True \
                            DATA_LOADER.NUM_WORKERS 8 \
                            AVA.FULL_TEST_ON_VAL True \
                            AVA.VAL_NO_CENTRE_CROP True \
                            OUTPUT_DIR evaluation/
```

### Train

Load the pretrained ST-Activity2Vec model and finetune with HAKE-AVA dataset:

```
# Set TRAIN.VAL_ONLY to False for training and evaluate per epoch.
python -u tools/run_net.py --cfg configs/HAKE/verb_base.yaml \
                            TRAIN.CHECKPOINT_FILE_PATH /your/checkpoint/path \
                            NUM_GPUS 8 \
                            TRAIN.PER_GPU_BS 4 \
                            TRAIN.ENABLE True \
                            TRAIN.VAL_ONLY False \
                            DATA_LOADER.NUM_WORKERS 8 \
                            AVA.FULL_TEST_ON_VAL False \
                            AVA.VAL_NO_CENTRE_CROP False \
                            OUTPUT_DIR training/
```

To simplify the whole command line, you may manually add the inputted options to the config file.