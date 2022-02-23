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
    height=224,
    width=224,
    transforms=['random_flip', 'random_crop'],
    batch_size_train=10,
    batch_size_test=10,
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)

model = build_model(
    name='alignedReID',
    num_classes=datamanager.num_train_pids,
    pretrained=True
)

model = model.cuda()
load_pretrained_weights(model,"./log/aligned/aligned_market/model.pth.tar-80")

optimizer = build_optimizer(
    model,
    optim='adam',
    lr=0.00001
)

scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[40],
    gamma=0.1
)

engine = Engine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion='aligned'
)

engine.run(
    save_dir='log/aligned',
    max_epoch=100,
    eval_freq=10,
    print_freq=20,
    test_only=False,
    save_name='aligned_market',
    normalize_feature=True,
    rerank=True,
    start_epoch=80
)