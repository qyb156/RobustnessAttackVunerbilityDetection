# Paper List for Robustness Attack of Deep Code Models

<!-- omit in toc -->

## Contents

- [Paper List for In-context Learning](#paper-list-for-in-context-learning)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Keywords Convention](#keywords-convention)
  - [Papers](#papers)
    - [Survey](#survey)
    - [Model Warmup for ICL](#model-warmup-for-icl)
    - [Prompt Tuning for ICL](#prompt-tuning-for-icl)
    - [Analysis of ICL](#analysis-of-icl)
      - [Influence Factors for ICL](#influence-factors-for-icl)
      - [Working Mechanism of ICL](#working-mechanism-of-icl)
    - [Evaluation and Resources](#evaluation-and-resources)
    - [Application](#application)
    - [Problems](#problems)
    - [Challenges and Future Directions](#challenges-and-future-directions)
  - [Blogs](#blogs)
  - [How to contribute?](#how-to-contribute)
  - [Reference](#reference)

## Introduction

This is a paper list (working in progress) about **Robustness Attack of Deep Code Models**

<!-- , for the following paper:

> [**A Survey for In-context Learning**](https://arxiv.org/abs/2301.00234),
> Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, Zhifang Sui.
> *arXiv preprint ([arXiv 2301.00234](https://arxiv.org/abs/2301.00234))* -->

## Papers

### Survey

1. **A Survey for In-context Learning**. ![](https://img.shields.io/badge/ICL_survey-DCE7F1)

   *Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, Zhifang Sui*.  [[pdf](https://arxiv.org/abs/2301.00234)], 2022.12, ![](https://img.shields.io/badge/arxiv-FAEFCA)

### Blackbox Attack

1. **Towards Robustness of Deep Program Processing Models -- Detection, Estimation and Enhancement**. ![](https://img.shields.io/badge/MetaICL-DCE7F1)

   *Sewon Min, Mike Lewis, Luke Zettlemoyer, Hannaneh Hajishirzi*.  [[pdf](https://arxiv.org/abs/2110.15943)], [[project]([https://github.com/facebookresearch/metaicl](https://github.com/SEKE-Adversary/CARROT))], 2023.10, 



### Evaluation and Resources

This section contains the pilot works that might contributes to the evaluation or resources of ICL.

1. **Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models**. ![](https://img.shields.io/badge/BigBench-DCE7F1)

   *Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt et. al.*.  [[pdf](https://arxiv.org/abs/2206.04615)], [[project](https://github.com/google/BIG-bench)], 2022.06, ![](https://img.shields.io/badge/conference-FAEFCA)
   ![](https://img.shields.io/badge/evaluation-EAD8D9) ![](https://img.shields.io/badge/large_scale-D8D0E1)
2. **SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Task**. ![](https://img.shields.io/badge/natural_instructions-DCE7F1)

   *Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit et. al.*.  [[pdf](https://arxiv.org/abs/2204.07705)], [[project](https://github.com/allenai/natural-instructions)], 2022.04, ![](https://img.shields.io/badge/EMNLP2022-FAEFCA)
    ![](https://img.shields.io/badge/instruction_tuning-D8D0E1)
3. **Language Models are Multilingual Chain-of-Thought Reasoners**.

   *Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei*.  [[pdf](https://arxiv.org/abs/2210.03057)], 2022.10, ![](https://img.shields.io/badge/conference-FAEFCA)
   ![](https://img.shields.io/badge/evaluation-EAD8D9) ![](https://img.shields.io/badge/multilingual-D8D0E1)

   - evaluate the reasoning abilities of large language models in multilingual settings, introduce the Multilingual Grade School Math (MGSM) benchmark, by manually translating 250 grade-school math problems from the GSM8K dataset into ten typologically diverse languages.
4. **Instruction Induction: From Few Examples to Natural Language Task Descriptions**. ![](https://img.shields.io/badge/Instruction_Induction-DCE7F1)

   *Or Honovich, Uri Shaham, Samuel R. Bowman, Omer Levy*.  [[pdf](https://arxiv.org/abs/2205.10782)], [[project](https://github.com/orhonovich/instruction-induction)], 2022.05, ![](https://img.shields.io/badge/conference-FAEFCA)
   ![](https://img.shields.io/badge/evaluation-EAD8D9) ![](https://img.shields.io/badge/learn_task_instructions-D8D0E1)

   - how to learn task instructions from input output demonstrations
5. **Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought**2022.10.3  ![](https://img.shields.io/badge/New-EAD8D9)
6. **What is Not in the Context? Evaluation of Few-shot Learners with Informative Demonstrations** 2212.01692.pdf (arxiv.org)  ![](https://img.shields.io/badge/New-EAD8D9)
7. **Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor**.

   *Or Honovich, Thomas Scialom, Omer Levy, Timo Schick*. [[pdf](https://arxiv.org/pdf/2212.09689.pdf)], [[project](https://github.com/orhonovich/unnatural-instructions)], 2022.12, ![](https://img.shields.io/badge/arxiv-FAEFCA)    ![](https://img.shields.io/badge/New-EAD8D9)
8. **Self-Instruct: Aligning Language Model with Self Generated Instructions**.

   *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi*. [[pdf](https://arxiv.org/pdf/2212.10560.pdf)], [[project](https://github.com/yizhongw/self-instruct)], 2022.12, ![](https://img.shields.io/badge/arxiv-FAEFCA)    ![](https://img.shields.io/badge/New-EAD8D9)
9. **The Flan Collection: Designing Data and Methods for Effective**.

   *Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, Adam Roberts*. [[pdf](https://arxiv.org/pdf/2301.13688.pdf)], [[project](https://github.com/google-research/FLAN/tree/main/flan/v2)], 2023.1, ![](https://img.shields.io/badge/arxiv-FAEFCA)    ![](https://img.shields.io/badge/New-EAD8D9)

### Application

This section contains the pilot works that expands the application of ICL.

1. **Meta-learning via Language Model In-context Tuning**.

   *Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, He He*.  [[pdf](https://arxiv.org/abs/2110.07814)], [[project](https://github.com/yandachen/in-context-tuning)], 2021.10, ![](https://img.shields.io/badge/ACL2022-FAEFCA)
   ![](https://img.shields.io/badge/application-EAD8D9) ![](https://img.shields.io/badge/Meta-learning-D8D0E1)

2. **Does GPT-3 Generate Empathetic Dialogues? A Novel In-Context Example Selection Method and Automatic Evaluation Metric for Empathetic Dialogue Generation**.

   *Young-Jun Lee, Chae-Gyun Lim, Ho-Jin Choi*.  [[pdf](https://aclanthology.org/2022.coling-1.56/)], 2022.10, ![](https://img.shields.io/badge/COLING2022-FAEFCA)
   ![](https://img.shields.io/badge/application-EAD8D9) ![](https://img.shields.io/badge/dialogue_generation-D8D0E1)

3. **In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models**. ![](https://img.shields.io/badge/ICL_Distillation-DCE7F1)

   *Yukun Huang, Yanda Chen, Zhou Yu, Kathleen McKeown*.  [[pdf](https://arxiv.org/abs/2212.10670)], 2022.12, ![](https://img.shields.io/badge/conference-FAEFCA)
   ![](https://img.shields.io/badge/challenge-EAD8D9) ![](https://img.shields.io/badge/distillation-D8D0E1)

4. **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions**![](https://img.shields.io/badge/New-EAD8D9)

5. **Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-shot In-Context Learner**.

   *Hyunsoo Cho, Hyuhng Joon Kim, Junyeob Kim, Sang-Woo Lee, Sang-goo Lee, Kang Min Yoo, Taeuk Kim*. [[pdf](https://arxiv.org/abs/2212.10873)], 2022.12, ![](https://img.shields.io/badge/AAAI2023-FAEFCA)
   ![](https://img.shields.io/badge/application-EAD8D9) ![](https://img.shields.io/badge/linear_probing-D8D0E1)

6. **In-Context Learning Unlocked for Diffusion Models**
   *Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou*. [[pdf]](https://arxiv.org/abs/2305.01115), [[project]](https://github.com/Zhendong-Wang/Prompt-Diffusion), 2023.5, ![](https://img.shields.io/badge/arxiv-FAEFCA)
   
7. **Molecule Representation Fusion via In-Context Learning for Retrosynthetic Plannings**
   *Songtao Liu, Zhengkai Tu, Minkai Xu, Zuobai Zhang, Lu Lin, Rex Ying, Jian Tang, Peilin Zhao, Dinghao Wu*. [[pdf]](https://arxiv.org/pdf/2209.15315.pdf), [[project]](https://github.com/SongtaoLiu0823/FusionRetro), 2023.5, ![](https://img.shields.io/badge/arxiv-FusionRetro)

This section contains the pilot works that points out the problems of ICL.

1. **The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design** ![](https://img.shields.io/badge/New-EAD8D9). ![](https://img.shields.io/badge/knn_Pretraining-DCE7F1)

   *Yoav Levine, Noam Wies, Daniel Jannai, Dan Navon, Yedid Hoshen, Amnon Shashua*.  [[pdf](https://arxiv.org/abs/2110.04541)], 2021.10, ![](https://img.shields.io/badge/ICLR2022-FAEFCA)
   ![](https://img.shields.io/badge/problem-EAD8D9) ![](https://img.shields.io/badge/knn_Pretraining-D8D0E1)

### Challenges and Future Directions

This section contains the pilot works that might contributes to the challenges and future directions of ICL.

## Blogs

[SEO is Dead, Long Live LLMO](https://jina.ai/news/seo-is-dead-long-live-llmo/)

[How does in-context learning work? A framework for understanding the differences from traditional supervised learning](http://ai.stanford.edu/blog/understanding-incontext/)

[Extrapolating to Unnatural Language Processing with GPT-3's In-context Learning: The Good, the Bad, and the Mysterious](http://ai.stanford.edu/blog/in-context-learning/)

[More Efficient In-Context Learning with GLaM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

## Open-source Toolkits
**OpenICL**
[[pdf](https://arxiv.org/abs/2303.02913)], [[project](https://github.com/Shark-NLP/OpenICL)], 2022.03

OpenICL provides an easy interface for in-context learning, with many state-of-the-art retrieval and inference methods built in to facilitate systematic comparison of LMs and fast research prototyping. Users can easily incorporate different retrieval and inference methods, as well as different prompt instructions into their workflow.

## Contribution

Please feel free to contribute and promote your awesome work or other related works here! 
If you recommend related works on ICL or make contributions on this repo, please provide your information (name, homepage) and we will add you to the contributor list😊.

### Contributor list 
We thank [Damai Dai](https://scholar.google.com/citations?user=8b-ysf0NWVoC&hl=zh-CN&oi=ao), [Qingxiu Dong](https://dqxiu.github.io/), [Lei Li](https://leili.site/), [Ce Zheng](https://scholar.google.com/citations?user=r7qFs7UAAAAJ&hl=zh-CN&oi=ao), [Shihao Liang](https://pooruss.github.io/-lshwebsite/), [Li Dong](http://dong.li/), [Siyin Wang](https://sinwang20.github.io/), [Po-Chuan Chen](https://jacksonchen1998.github.io/) for their repo contribution and paper recommendation.


<!-- ## Citations -->
## Reference

<!-- Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows. -->
Some papers are discussed in the following paper:

```
@misc{dong2022survey,
      title={A Survey for In-context Learning}, 
      author={Qingxiu Dong and Lei Li and Damai Dai and Ce Zheng and Zhiyong Wu and Baobao Chang and Xu Sun and Jingjing Xu and Lei Li and Zhifang Sui},
      year={2022},
      eprint={2301.00234},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



