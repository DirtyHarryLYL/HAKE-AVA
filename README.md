# HAKE-AVA
Fine-grained Spatio-Temporal Activity Understanding based on AVA videos. 
A part of the [HAKE](http://hake-mvig.cn) project.

## Annotation Diagram

<div align=center>
<img src="figs\hake-ava.png" width="800" />
</div>

### HAKE-AVA-PaSta (Body part states in AVA) 
HAKE-AVA contains the human body part states (PaSta) annotations upon AVA (v2.1 & 2.2) and covers all the labeled human instances. PaSta (Part State) describes the action states of 10 human body parts, i.e., head, arms, hands, hip, legs, and feet.

For the procedure of preparing HAKE-AVA-PaSta dataset, please refer to [DATASET.md](./DATASET.md).

**[ST-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-AVA/tree/ST-Activity2Vec)**: A PaSta-based activity understanding model. Its overall pipeline is same with the image-based [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) except using different feature backbone (ResNet -> SlowFast). We also provide the weights pretrained on Kinetics-600 and finetuned on HAKE-AVA.

**[CLIP-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/CLIP-Activity2Vec)**: We also release a CLIP-based human body part states recognizer in [CLIP-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/CLIP-Activity2Vec)!

Besides, in our another work, we also annotate all the interactive objects in AVA 2.2 videos:

### HAKE-DIO (Object boxes in AVA)
HAKE-DIO contains the bounding box (290 K) and object class (1,000+) annotations of all the interacive objects in AVA videos (v2.2), according to the labeled humans in AVA v2.2 performing Human-Object Interactions (HOI, 51 classes). 

For more details, please refer to this [[branch]](https://github.com/DirtyHarryLYL/HAKE-AVA/tree/DIO) and [[Paper]](https://arxiv.org/abs/2211.07501).

### Joint version: HAKE-AVA-PaSta + HAKE-DIO
We also provide a joint version combining the human body part states and interactive object boxes in one file, as shown in the above figure. Please refer to [[this file]](https://github.com/DirtyHarryLYL/HAKE-AVA/blob/DIO/DATASET.md).

## Citation
 If you find our works useful, please consider citing:
```
@article{li2022hake,
  title={HAKE: A Knowledge Engine Foundation for Human Activity Understanding},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Qiu, Zuoyu and Xu, Liang and Xu, Yue and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:2202.06851},
  year={2022}
}
@article{li2022discovering,
  title={Discovering a Variety of Objects in Spatio-Temporal Human-Object Interactions},
  author={Li, Yong-Lu and Fan, Hongwei and Qiu, Zuoyu and Dou, Yiming and Xu, Liang and Fang, Hao-Shu and Guo, Peiyang and Su, Haisheng and Wang, Dongliang and Wu, Wei and Lu, Cewu},
  journal={arXiv preprint arXiv:2211.07501},
  year={2022}
}
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

## Main Project: HAKE (Human Activity Knowledge Engine)

For more details please refer to HAKE website http://hake-mvig.cn.

- **HAKE-Image** (CVPR'18/20): Human body part state labels in images. [HAKE-HICO](https://github.com/DirtyHarryLYL/HAKE#hake-hico-for-image-level-hoi-recognition), [HAKE-HICO-DET](https://github.com/DirtyHarryLYL/HAKE#hake-hico-det-for-instance-level-hoi-detection), [HAKE-Large](https://github.com/DirtyHarryLYL/HAKE#hake-large-for-instance-level-action-understanding-pre-training), [Extra-40-verbs](https://github.com/DirtyHarryLYL/HAKE#extra-40-verb-categories).
- **HAKE-AVA**: Human body part state labels in videos from AVA dataset. [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA).
- **[HAKE-A2V](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec)** (CVPR'20): Activity2Vec, a general activity feature extractor based on HAKE data, converting a human (box) to a fixed-size vector, PaSta and action scores.
- **[HAKE-Action-TF](https://github.com/DirtyHarryLYL/HAKE-Action), [HAKE-Action-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch)** (CVPR'19/20/22, NeurIPS'20, TPAMI'21, ECCV'22): SOTA action understanding methods and the corresponding HAKE-enhanced versions ([TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network), [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)), [IF](https://github.com/Foruck/Interactiveness-Field), [mPD](https://github.com/Foruck/OC-Immunity), [PartMap](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness)).
- **HAKE-3D** (CVPR'20): 3D human-object representation for action understanding ([DJ-RN](https://github.com/DirtyHarryLYL/DJ-RN)).
- **HAKE-Object** (CVPR'20, TPAMI'21): object knowledge learner to advance action understanding ([SymNet](https://github.com/DirtyHarryLYL/SymNet)).
- [**Halpe**](https://github.com/Fang-Haoshu/Halpe-FullBody): a joint project under [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and [HAKE](http://hake-mvig.cn), full-body human keypoints (body, face, hand, 136 points) of 50,000 HOI images.
- [**HOI Learning List**](https://github.com/DirtyHarryLYL/HOI-Learning-List): a list of recent HOI (Human-Object Interaction) papers, code, datasets and leaderboard on widely-used benchmarks. Hope it could help everyone interested in HOI.

#### **News**: (2022.11.19) We release the interactive object bounding boxes & classes in the interactions within AVA dataset (2.1 & 2.2)! [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA), [[Paper]](https://arxiv.org/abs/2211.07501).

(2022.07.29) Our new work PartMap is released! [Paper](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness/blob/main), [Code](https://github.com/DirtyHarryLYL/HAKE-Action-Torch)

(2022.04.23) Two new works on HOI learning are releassed! [Interactiveness Field](https://arxiv.org/abs/2204.07718) (CVPR'22) and a new HOI metric [mPD](https://arxiv.org/abs/2202.09492) (AAAI'22).

(2022.02.14) We release the human body part state labels based on AVA: [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA) and [HAKE 2.0 paper](https://arxiv.org/abs/2202.06851).

(2021.10.06) Our extended version of [SymNet](https://github.com/DirtyHarryLYL/SymNet) is accepted by TPAMI! Paper and code are coming soon.

(2021.2.7) Upgraded [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) is released! Images/Videos --> human box + ID + skeleton + part states + action + representation. [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing)
<p align='center'>
    <img src="https://github.com/DirtyHarryLYL/HAKE-Action-Torch/blob/Activity2Vec/demo/a2v-demo.gif", height="400">
</p>

(2021.1.15) Our extended version of [TIN (Transferable Interactiveness Network)](https://arxiv.org/abs/2101.10292) is accepted by TPAMI!

(2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) ([Paper](https://arxiv.org/abs/2010.16219)) in NeurIPS'20 is released!

(2020.6.16) Our larger version [HAKE-Large](https://github.com/DirtyHarryLYL/HAKE#hake-large-for-instance-level-hoi-detection) (>122K images, activity and part state labels) and [Extra-40-verbs](https://github.com/DirtyHarryLYL/HAKE#extra-40-verb-categories) (40 new actions) are released!

## TODO
- [X] ava 2.1 pasta annotation download manuscript (ava data, our labels, basic structure)
- [X] dio annotation download manuscript, 
- [ ] DIO basic structure
- [X] fusing dio and hake-ava data and labels
