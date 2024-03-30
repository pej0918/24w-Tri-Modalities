# Training 
```
python train.py --use_softmax True --use_cls_token False --exp 
```
```
CUDA_VISIBLE_DEVICES='2,3' python -m torch.distributed.launch --nproc_per_node=2 train.py
```

# Dataset 
MSR VTT 
- Original dataset: [download](https://www.dropbox.com/sh/bd75sz4m734xs0z/AADbN9Ujhn6FZX12ulpNWyR_a?dl=0)
- Our dataset: [download](https://drive.google.com/drive/folders/1JsGZKp3ZAoC7w2XaOkZp4TnQ0GwGwUtU?usp=sharing)
