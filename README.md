# LGSearch_DDGAN_PyTorch 

Official Implementation of Joint Local and Global Search for Visual Tracking with Adversarial Learning. 

BMVC-2019: Learning Target-aware Attention for Robust Tracking with Conditional Adversarial Network, Xiao Wang, Tao Sun,  Rui Yang, Bin Luo [[Project](https://sites.google.com/view/globalattentiontracking/home)] [[Paper](https://bmvc2019.org/wp-content/uploads/papers/0562-paper.pdf)] [[Supplement](https://bmvc2019.org/wp-content/uploads/papers/0562-supplementary.pdf)] [[Poster](https://drive.google.com/file/d/1BYxTYnxYKjPv8Hu7EjwzgLlcbCjNg-Z2/view)]  


Journal Extension: Tracking by Joint Local and Global Search: A Target-aware Attention based Approach, Xiao Wang, Jin Tang, Bin Luo, Yaowei Wang, Yonghong Tian, and Feng Wu, IEEE TNNLS 2021 [[Paper](https://arxiv.org/abs/2106.04840)] [[Video](https://www.youtube.com/watch?v=uu00QIL7tjo&ab_channel=XiaoWang)] [[Slides](https://drive.google.com/file/d/1IgD4msLVVXjR5lk1sFiJN9rlfvIuvhcR/view)] 

## Abstract 
Tracking-by-detection is a very popular framework for single object tracking which attempts to search the target object within a local search window for each frame. Although such local search mechanism works well on simple videos, however, it makes the trackers sensitive to extremely challenging scenarios, such as heavy occlusion and fast motion. In this paper, we propose a novel and general target-aware attention mechanism (termed TANet) and integrate it with tracking-by-detection framework to conduct joint local and global search for robust tracking. Specifically, we extract the features of target object patch and continuous video frames, then we concatenate and feed them into a decoder network to generate target-aware global attention maps. More importantly, we resort to adversarial training for better attention prediction. The appearance and motion discriminator networks are designed to ensure its consistency in spatial and temporal views. In the tracking procedure, we integrate the target-aware attention with multiple trackers by exploring candidate search regions for robust tracking. Extensive experiments on both short-term and long-term tracking benchmark datasets all validated the effectiveness of our algorithm. 


## Tracking Framework 
![rgbt_car10](https://github.com/wangxiao5791509/LGSearch_DDGAN_PyTorch/blob/master/pipeline.png) 



## Inference
1. You can directly generate all the attention images for your testing dataset, for example, [[GOT-10K](http://got-10k.aitestunion.com/)] or [[TNL2K](https://sites.google.com/view/langtrackbenchmark/)].  
~~~
python test_got10k.py
~~~

2. You can also integrate the code into your own tracker, and conduct local-global search only when needed. Our pre-trained model on the GOT-10K training subset is available [**here**](https://drive.google.com/file/d/1g3TsL3_qajKdL0bUp1iBwoiq8USHUpGZ/view?usp=sharing). You can use it to predict gloabl attention for your tracker. 



## Cite 

If you find this paper useful for your research, please consider citing our paper:
~~~
@inproceedings{wang2019GANTrack,
  title={Learning Target-aware Attention for Robust Tracking with Conditional Adversarial Network},
  author={Wang, Xiao and Sun, Tao and Yang, Rui and Luo, Bin},
  booktitle={30TH British Machine Vision Conference},
  year={2019}
} 

@inproceedings{wang2021ganTANetTrack,
  title={Tracking by Joint Local and Global Search: A Target-aware Attention based Approach},
  author={Wang, Xiao and Tang, Jin and Luo, Bin and Wang, Yaowei and Tian, Yonghong and Wu, Feng },
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
} 
~~~

If you have any questions, please contact me via email: wangxiaocvpr@foxmail.com. 



