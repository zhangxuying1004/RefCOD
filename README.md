# <p align=center>`Referring Camouflaged Object Detection `</p>
> **Authors:**
> [Xuying Zhang](https://zhangxuying1004.github.io/),
> [Bowen Yin](http://yinbowen-chn.github.io/),
> [Zheng Lin](https://www.lin-zheng.com/),
> [Qibin Hou](https://houqb.github.io/),
> [Deng-Ping Fan](https://dengpingfan.github.io/), 
> [Ming-Ming Cheng](https://mmcheng.net/).

This repo contains the official dataset and source code of the paper [_Referring Camouflaged Object Detection_](https://arxiv.org/pdf/2306.07532.pdf).   
In this paper, we consider the problem of referring camouflaged object detection (Ref-COD), a new task that aims to segment specified
camouflaged objects based on a small set of referring images with salient target objects. 
  
<p align="center">
    <img src="figs/refcod.png" width="450"/> <br />
    <em> 
    Fig. 1: Visual comparison between the standard COD and our Ref-COD.
    Given an image containing multiple camouflaged objects, the COD
    model tends to find all possible camouflaged objects that are blended
    into the background without discrimination, while the Ref-COD model
    attempts to identify the camouflaged objects under the condition of a set
    of referring images.
    </em>
</p>

**Note**: I will upload the codes including the embedding process of the common representations of target objects, the attribution evaluation of different COD / Ref-COD methods, etc, within several weeks.
And you can first use my processed representations at the below dataset link if you are interested in our Ref-COD topic.

## Environment setup
``` 
conda env create -f environment.yml
conda activate refcod
```

## Get Start
**1. Dataset.**
<p align="center">
    <img src="figs/r2c7k.png" width="970"/> <br />
    <em>
      Fig. 2. Examples from our R2C7K dataset. Note that the camouflaged objects in Camo-subset are masked with their annotations in orange.
    </em>
</p>

- Download our ensembled [R2C7K](https://pan.baidu.com/s/13wRyW81TIPjVCj9GP7o4BQ) dataset with access code ```2023``` on Baidu Netdisk.  
```   
├── R2C7K  
    ├── Camo  
        ├── train                # training set of camo-subset with 64 categories.  
        └── test                 # tesing set of camo-subset with 64 categories.  
    ├── Ref          
        ├── Images               # all images of ref-subset with 64 categories.
        ├── RefFeat_ICON-R       # all object representations of ref-subset with 64 categories.  
        └── Saliency_ICON-R      # all foreground maps of ref-subset with 64 categories.  
```
- Update the 'data_root' param with your R2C7K location in ```train.py```, ```infer.py``` and ```test.py```.

**2. Framework**
<p align="center">
    <img src="figs/r2cnet.png" width="950"/> <br />
    <em>
      Fig. 3. Overall architecture of our R2CNet framework, which is composed of two branches, i.e., reference branch in green and segmentation branch
in orange. In the reference branch, the common representation of a specified object from images is obtained by masking and pooling the visual
features with the foreground map generated by a SOD network. In the segmentation branch, the visual features from the last three layers of the
encoder are employed to represent the given image. Then, these two kinds of feature representations are fused and compared in the well-designed
RMG module to generate a mask prior, which is used to enrich the visual feature among different scales to highlight the camouflaged targets in our
RFE module. Finally, the enriched features are fed into the decoder to generate the final segmentation map. DSF: Dual-source Information Fusion, MSF: Multi-scale Feature Fusion, TM: Target Matching.
    </em>
</p>

**3. Train.**
```
python train.py --model_name r2cnet --gpu_id 0
```

**4. Infer.**
- Download the pre-trained [r2cnet.pth](https://pan.baidu.com/s/1daqxGTy120JondOIvCAEOw) checkpoints with access code ```2023``` on Baidu Netdisk.
- Put the checkpoint file on './snapshot/saved_models/'.
- Run ```python infer.py``` to generate the foreground maps of R2CNet.
- You can also directly refer to the predictions [R2CNet-Maps](https://pan.baidu.com/s/1unQQOn9w3rW9aWdnYf_zrA) with access code ```2023``` on Baidu Netdisk.

**5. Test.**
- Assert that the pre-trained [r2cnet.pth](https://pan.baidu.com/s/1daqxGTy120JondOIvCAEOw) checkpoint file has been placed in './snapshot/saved_models/'.
- Run ```python test.py``` to evaluate the performance of R2CNet.

**6. Ref-COD Benchmark Results.**
<p align="center">
    <em>
      Tab. 1. Comparison of the COD models with their Ref-COD counterparts. All models are evaluated on a NVIDIA RTX 3090 GPU. ‘R-50’: ResNet-50 [82],
      ‘E-B4’: EfficientNet-B4 [86], ‘R2-50’: Res2Net-50 [87], ‘R3
      -50’: Triple ResNet-50 [2]. ‘-Ref’: the model with image references composed of salient
      objects. ‘Attribute’: the attribute of each network, ‘Single-obj’: the scene of a single camouflaged object, ‘Multi-obj’: the scene of multiple
      camouflaged objects, ‘Overall’: all scenes containing camouflaged objects.
    </em>
    <img src="figs/benchmarks.png" width="1000"/> <br />
</p>

## Contact
For technical questions, feel free to contact [zhangxuying1004@gmail.com]() and [bowenyin@mail.nankai.edu.cn]().

## Citation
If our work is helpful to you or gives some inspiration to you, please star this project and cite our paper. Thank you!  
```
@article{zhang2023referring,
  title={Referring Camouflaged Object Detection},
  author={Zhang, Xuying and Yin, Bowen and Lin, Zheng and Hou, Qibin and Fan, Deng-Ping and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2306.07532},
  year={2023}
}
```

## Acknowlegement
This repo is mainly built based on [SINet-V2](https://github.com/GewelsJI/SINet-V2), [PFENet](https://github.com/dvlab-research/PFENet) and [MethodsCmp](https://github.com/lartpang/MethodsCmp). Thanks for their great work!

