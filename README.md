# Person_reID_pytorch

## Train
1. Load datamanager

```python
# For data transformation, change parameter in ImageDataManager
datamanager = ImageDataManager(
    root='dataset',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)
```
2. Build model, optimizer and lr_scheduler

```python
model = build_model(
    name='pcb',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = build_optimizer(
    model,
    optim='sgd',
    lr=0.1
)

#If you want decay lerning rate in specific epoch, give the epoch list as the `step size` parameter.
scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[30,50]
)
```

3. Build engine and run

```python
engine = Engine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

4. Run

```python
engine.run(
    save_dir='log/pcb',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False,
)
```
## Test
Apply steps 1 to 3 in the same way as the training process.
In step 4, change the `test_only` to `True`.
```python
# Make sure set `test_only=True`
# If you set `visrank=True`, The top 10 similar gallery images for a given query will be saved to a file.
engine.run(
    save_dir='log/osnet',
    max_epoch=60,
    eval_freq=10,
    print_freq=20,
    test_only=True,
    visrank=True,
    open_layers='classifier'
)
```
---
## Datasets
- [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
- [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf)
- [DukeMTMC-reID](https://arxiv.org/abs/1701.07717)
---
## Models
- [PCB](https://arxiv.org/abs/1711.09349)
- [AlignedReID](https://arxiv.org/pdf/1711.08184v2.pdf)
- [MGN](https://arxiv.org/pdf/1804.01438)
