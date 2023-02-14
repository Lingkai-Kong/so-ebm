# SO-EBM
This repo contains our code for paper: 

End-to-End Stochastic Optimization with Energy-Based Model

Lingkai Kong, Jiaming Cui, Yuchen Zhuang, Rui Feng, B. Aditya Prakash, Chao Zhang

[[paper](https://arxiv.org/abs/2211.13837)] 

![SDE-Net](figure/overall.png)


## Training and Evaluation

Training with the two-stage model:
```
python two-stage.py --lr 0.001
```

Training with DFL:
```
python DFL.py --lr 0.0001
```
Evaluation for the baselines:
```
python test_baseline.py 
```
Training with SO-EBM
```
python so-ebm.py
```

Note that DFL and SO-EBM needs to first train the two-stage model as the pre-trained model.

## Citation
Please cite the following paper if you find this repo helpful. Thanks!
```
@inproceedings{kongend,
  title={End-to-end Stochastic Optimization with Energy-based Model},
  author={Kong, Lingkai and Cui, Jiaming and Zhuang, Yuchen and Feng, Rui and Prakash, B Aditya and Zhang, Chao},
  booktitle={Advances in Neural Information Processing Systems}
  year={2022}
}
```
```
@article{donti2017task,
  title={Task-based end-to-end model learning in stochastic optimization},
  author={Donti, Priya and Amos, Brandon and Kolter, J Zico},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```# so-ebm
