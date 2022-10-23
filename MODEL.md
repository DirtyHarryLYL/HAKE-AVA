## Pretrained Model of ST-Activity2Vec

We evaluate the performance of the pretrained model([link](https://1drv.ms/u/s!ArUVoRxpBphYguIbOhuhI_z1kY7x1w?e=Fi9W8W)) on **PaSta and action detection** tasks, and report the mAP results on each task. For AVA benchmark, we follow the default setting of ActivityNet, which chooses 60 classes from the original 80 classes of AVA, and only evaluates the detection performance on these classes. The results are provided below:

|  Task               | mAP       |
|  ----               | ----      |
|  PaSta: foot        | 24.83     |
|  PaSta: leg         | 25.73     |
|  PaSta: hip         | 38.48     |
|  PaSta: hand        | 8.69      |
|  PaSta: arm         | 21.74     |
|  PaSta: head        | 31.85     |
|  **PaSta: avg**     | **25.22** |
|  **Action**         | **28.82** |

Set the config TRAIN.CHECKPOINT_FILE_PATH as the path of pretrained model to use it.