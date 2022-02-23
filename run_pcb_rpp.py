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
    batch_size_train=10,
    batch_size_test=10,
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)

model = build_model(
    name='pcb',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)


model.convert_rpp()
load_pretrained_weights(model,"./log/pcb_rpp/pcb_rpp_market/model.pth.tar-10")
model = model.cuda()

optimizer = build_optimizer(
    model,
    optim='sgd',
    lr=0.01
)

scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=10,
    gamma=0.1
)

engine = Engine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
)

engine.run(
    save_dir='log/pcb_rpp',
    max_epoch=20,
    eval_freq=10,
    print_freq=10,
    test_only=False,
    save_name='pcb_rpp_market'
)