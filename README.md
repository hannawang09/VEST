<div align='center'>
<h1>Enabling Validation for Robust Few-Shot Recognition</h1>
	
<a href="https://hannawang09.github.io/" target="_blank">Hanxin Wang</a><sup>1,\*</sup>,
<a href="https://tian1327.github.io/" target="_blank">Tian Liu</a><sup>2,\*</sup>,
<a href="https://aimerykong.github.io/" target="_blank">Shu Kong</a><sup>1,3</sup>

<span><sup>1</sup>University of Macau,</span>
<span><sup>2</sup>Texas A&M University,</span>
<span><sup>3</sup>Institute of Collaborative Innovation</span>

<sup>*</sup>Equal contribution
 
<a href="https://arxiv.org/abs/2506.04713"><img src='https://img.shields.io/badge/arXiv-VEST-red' alt='Paper PDF'></a>
<a href="https://hannawang09.github.io/projects/vest/"><img src='https://img.shields.io/badge/Project_Page-VEST-green' alt='Project Page'></a>
</div>


We introduce <b>gF1</b>, a novel validation method that repurposes retrieved OOD data for checkpoint selection and hyperparameter tuning in few-shot recognition. We further integrate gF1 into <b>VEST</b> (<b>V</b>alidation-<b>E</b>nabled <b>S</b>tage-wise <b>T</b>uning), a stage-wise finetuning pipeline that improves both ID and OOD peformance.



<div align='center'>
    <img src='asset/gF1.png' alt='gF1' style="height:170px; width:auto;">
    &nbsp;&nbsp;&nbsp;
    <img src='asset/performance.png' alt='performance' style="height:170px; width:auto;">
</div>


## Environment Configuration

You can run the command below to set up the environment in an easy way:

```
# Create a Virtual Environment
conda create -n vest python=3.10
conda activate vest
# Install Dependencies
pip install -r requirements.txt
```



## Dataset Preparation 

Please follow the instructions in [DATASET.md](DATASETS.md) to prepare the datasets used in the experiments.



## Training and Testing
1. Update your data path and retrieved data path in `config.yml`.
2. Running script 
    - For **Validation-Enabled Stage-wise Tuning (VEST)**, use the following command:
    ```
    bash scripts/run_dataset_seed_VEST.sh imagenet [data_seed] [ft_top_X_block]
    
    # In our experiments, we PFT top-4 blocks on CLIP and top-1 blocks on DINOv2.
    ```
    - For **Partial Finetuning (PFT)**, use the following command:
    ```
    bash scripts/run_dataset_seed_PFT.sh imagenet [data_seed] [ft_top_X_block]
    ```

    > Note: The default model is CLIP. To finetune the DINOv2 model instead, please update the `model_cfg` in scripts.





## Demos
We provide demos of model training and evaluation. 

- See `PFT_demo.ipynb` for the details of **Partial Finetuning**.
- See `VEST_demo.ipynb` for the details of **Validation-Enabled Stage-wise Tuning**.
- See `VEST_dinov2_demo.ipynb` for the details of **Validation-Enabled Stage-wise Tuning** on vision foundation model DINOv2.




## Acknowledgments

Our code is built on [LCA-on-the-line(ICML'24)](https://github.com/ElvishElvis/LCA-on-the-line) and [SWAT(CVPR'25)](https://github.com/tian1327/SWAT).


## Citation

If you find our project useful, please consider citing:

```bibtex
@article{wang2025enabling,
    title={Enabling Validation for Robust Few-Shot Recognition}, 
    author={Wang, Hanxin and Liu, Tian and Kong, Shu},
    journal={arXiv preprint arXiv:2506.04713},
    year={2025}
}

@inproceedings{liu2025few,
  title={Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning},
  author={Liu, Tian and Zhang, Huixin and Parashar, Shubham and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
