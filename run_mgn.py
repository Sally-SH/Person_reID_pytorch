from data.datamanager import ImageDataManager
from models import build_model
from optim.optimizer import build_optimizer
from optim.scheduler import build_lr_scheduler
from engine.engine import Engine
from utils.torchtools import load_pretrained_weights

datamanager = ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=384,
    width=128,
    transforms=['random_flip'],
    batch_size_train=5,
    batch_size_test=10,
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)

model = build_model(
    name='mgn',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
)

model = model.cuda()
load_pretrained_weights(model,"./log/mgn/mgn_market/model.pth.tar-80")

optimizer = build_optimizer(
    model,
    optim='sgd',
    lr=0.01
)

scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[40, 60],
    gamma=0.1
)

engine = Engine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion='mgn'
)

engine.run(
    save_dir='log/mgn',
    max_epoch=80,
    eval_freq=10,
    print_freq=10,
    test_only=True,
    save_name='mgn_market',
    rerank=True
)