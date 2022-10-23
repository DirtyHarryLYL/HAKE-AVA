<!-- # HAKE-Video
A video-based HAKE sub-project.

## HAKE-AVA PaSta Annotations
HAKE-AVA contains the human body part states (PaSta) annotations upon AVA v2.1 and covers all the labeled human instances. 
PaSta describes the action states of 10 human body parts, i.e., head, arms, hands, hip, legs, and feet.

TODO: fig depiction of ava video and hoi labels, pasta labels

For data preparation, please see this [[description]](https://github.com/DirtyHarryLYL/HAKE-Video/blob/ST-Activity2Vec/DATASET.md). -->

# ST-Activity2Vec

A PaSta-based activity understanding model (using a simple linear activity classifier, w/o reasoing module currently). Its overall pipeline is the same as the image-based [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) except using a different backbone (ResNet -> SlowFast) for videos. We also provide the weights pretrained on Kinetics-600 and finetuned on HAKE-AVA.

<!-- # ST-Activity2Vec (ST-A2V) -->
<!-- General human activity feature extractor and human PaSta (part states) detector based on HAKE-A2V.
It works like an ImageNet/COCO pre-trained backbone, which aims at extracting multi-modal activity representation for downstream tasks like VQA, captioning, clustering, etc.  -->

### Paper
PaStaNet: Toward Human Activity Knowledge Engine (CVPR'20), [Project](http://hake-mvig.cn), [Paper](https://arxiv.org/abs/2004.00945), [Code-TF](https://github.com/DirtyHarryLYL/HAKE-Action), [Code-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch), [Code-Video](https://github.com/DirtyHarryLYL/HAKE-Video)

Yong-Lu Li, Liang Xu, Xinpeng Liu, Xijie Huang, Yue Xu, Shiyi Wang, Hao-Shu Fang, Ze Ma, Mingyang Chen, Cewu Lu.

### Pipeline: 
Video --> human detection --> PaSta classification --> Activity CLassification

HAKE-A2V (tracklet, person box) = PaSta detection (87 classes) + Activity detection (60 classes)

More details can be found in [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) and [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec).

<!-- <p align='center'>
    <img src="demo/a2v-demo.gif", height="400">
</p> -->

<!-- ## Full demo: [[YouTube]](https://t.co/hXiAYPXEuL?amp=1), [[bilibili]](https://www.bilibili.com/video/BV1s54y1Y76s)
#### Contents in demo, [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing)
<!-- - human ID & box & skeleton -->
<!-- - body part box & states -->
<!-- - human actions -->

## Installation
 To install the overall framework of ST-Activity2Vec, please follow [INSTALL.md](./INSTALL.md).

## Dataset
 For the procedure of preparing HAKE-AVA dataset for ST-Activity2Vec, please refer to [DATASET.md](./DATASET.md).

## Pretrained Models
 For the download links of the pretrained ST-Activity2Vec models, please refer to [MODEL.md](./MODEL.md).
 
## Getting Started
 To start your journey with ST-Activity2Vec, please refer to [GETTING_STARTED.md](./GETTING_STARTED.md).

## Contributors
 This branch is contributed by Hongwei Fan ([@hwfan](https://github.com/hwfan)), Yiming Dou([@Dou-Yiming](https://github.com/Dou-Yiming)), Yong-Lu Li ([@DirtyHarryLYL](https://github.com/DirtyHarryLYL)). Please contact them if there are any problems.
 
## Citation
 If you find our works useful, please consider citing:
```
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

