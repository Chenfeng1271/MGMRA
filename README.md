# **[Memory Regulation and Alignment toward Generalizer RGB-Infrared Person Re-identification](https://arxiv.org/abs/2109.08843)**


### Highlights
- The learned coarse-to-fine prototypes can consistently provide domain-level semantic templets with various granularity, meeting the requirement for multi-level semantic alignment.
- Our proposed MG-MRA boosts the performance of baseline and existing state of the arts, e.g., AGW  and HCT  by a large margin with limited consumptions. We achieve a new state-of-the-art on RegDB  and SYSU-MM01 with 94.59%/88.18% and 72.50%/68.94% Rank1/mAP respectively.

### Method
![image-20210909100353763](20210918132449.png)

### Results

![image-20210909100353763](image-20210909100353763.png)


### Usage
Our code extends the pytorch implementation of Cross-Modal-Re-ID-baseline in [Github](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). Please refer to the offical repo for details of data preparation.

### Training

Train original HCT method for RegDB by

```bash
python train_HCT.py --dataset regdb --lr 0.1 --gpu 0 --batch-size 8 --num_pos 4
```

Train a SG-MRA for RegDB by
```bash
python train_SGMRA.py --dataset regdb --lr 0.1 --gpu 0 --batch-size 8 --num_pos 4
```

Train a MG-MRA for RegDB by

```bash
python train_MGMRA.py --dataset regdb --lr 0.1 --gpu 0 --batch-size 8 --num_pos 4
```

Train a model for SYSU-MM01 by

```bash
python train_MGMRA.py --dataset sysu --lr 0.1 --batch-size 6 --num_pos 8 --gpu 0
```

**Parameters**: More parameters can be found in the manuscript and code.

### Reference
```
@article{chen2021memory,
  title={Memory Regulation and Alignment toward Generalizer RGB-Infrared Person},
  author={Chen, Feng and Wu, Fei and Wu, Qi and Wan, Zhiguo},
  journal={arXiv preprint arXiv:2109.08843},
  year={2021}
}

@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```
