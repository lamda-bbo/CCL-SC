# **Confidence-aware Contrastive Learning for Selective Classification**

This is the official implementation of the paper **Confidence-aware Contrastive Learning for Selective Classification**. 

Based on our theoretical analysis, in this work, we propose a novel confidence-aware contrastive learning method for selective classification that explicitly improve the selective classification model at the feature level.

## Requirements

```
matplotlib==3.5.3
numpy==1.21.6
Pillow==9.4.0
progress==1.6
scipy==1.10.0
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
```

## Training and Evaluation

**Run CCL-SC:**

```
bash run_${dataset}_csc.sh
```

## Citation

```
@inproceedings{
    CCL-SC,
    title={Confidence-aware Contrastive Learning for Selective Classification},
    author={Yu-Chang Wu, Shen-Huan Lyu, Haopu Shang, Xiangyu Wang, Chao Qian},
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```

## Acknowledgement

This code is based on the official code base of [SAT+EM](https://github.com/BorealisAI/towards-better-sel-cls) (which is based on  the official code base of [Deep Gambler](https://github.com/Z-T-WANG/NIPS2019DeepGamblers) and  [Self-Adaptive Training](https://github.com/LayneH/SAT-selective-cls)).

