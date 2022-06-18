from py_lightning_modules.py_lightning_dataModule import DataModule_lightning
from py_lightning_modules.py_lightning_models import  lightning_module
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
import argparse 
from mup.coord_check import get_coord_data, plot_coord_data
from mup import get_shapes, make_base_shapes, set_base_shapes
from convNet_architectures.resnet import ResNet18, ResNet34, ResNet50
from convNet_architectures import resnet

import wandb
import numpy as np
import torch
from mup import init
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode



##**************************************************************************
## Setting Some of Hyperparameters 
##**************************************************************************

parser= argparse.ArgumentParser()
## Dataset Define
parser.add_argument('--dataloader_type', type=str, default='Standard_dataloader', choices=['ffcv_loader', 'others'])
parser.add_argument('--dataset_name', type=str, default='CIFAR100')
parser.add_argument('--dataset_mean', type=list, default=None)
parser.add_argument('--dataset_std', type=list, default=None)
parser.add_argument('--dataset_length', type=int, default=50000, required=False)

parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--data_dir', type=str, default='/data/downstream_datasets/CIFAR100/')
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)

## Training Hyperparameters
parser.add_argument('--seed', type=int, default=100,help='random seed')
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet34'])
parser.add_argument('--width_mult', type=float, default=1)
parser.add_argument('--m_transfer_enable', type=bool, default=True)
parser.add_argument('--metric', type=str, default='Accuracy')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optimizer_type', default='MuAdam', )#choices=['SGD', 'Adam','AdamW' 'MuSGD', 'MuAdam', 'MuAdamW'])
parser.add_argument('--init_type', default="uniform")# Choices=kaiming_normal, kaiming_uniform, normal, uniform, xavier_normal, xavier_uniform
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument("--lr_scheduler",type= str, default="step",)# choices=['step', 'reduce_plateau', 'linear', 'cosineAnnealing', 'ConsineAnl_warmup'], help="learning rate schedule")
parser.add_argument("--steps", type= list, default=[30, 50, 80, 120],  help="learning rate schedule")#[30,50,90, 120]
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gpus', type=list, default=[0,1])

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument("--RandAug", type=list, default=[1, 9], help="Enable Random Augmentation or Not [1, 9]", )
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--save_base_shapes', type=str, default='',
                    help='file location to save base shapes at')
parser.add_argument('--load_base_shapes', type=str, default='/data/efficient_training/resnet18.bsh',
                    help='file location to load base shapes from')

## Visualization and Debug Setting
parser.add_argument("--method", type=str, default="mtransfer_all_layers")
parser.add_argument("--job_type", type=str, default="hyperparameter tuning")

parser.add_argument('--coord_check', type =bool, default=False, #action='store_true',
                    help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
parser.add_argument('--coord_check_nsteps', type=int, default=4,
                    help='Do coord check with this many steps.')
parser.add_argument('--coord_check_nseeds', type=int, default=1,
                    help='number of seeds for coord check')
parser.add_argument('--plotdir', type=str, default='/data/efficient_training/efficient_training_neural_Nets/m_transfere_coord_check',
                    help='The path to store coordinate check image')
                                     

args = parser.parse_args()


# sweep_config = {
#   "method":  "bayes",   #"random",   # Random search
#   "metric": {           # We want to maximize val_acc
#       "name": "val_loss",
#       "goal": "minimize"
#   },
#   "parameters": {
#         # "n_layer_1": {
#         #     # Choose from pre-defined values
#         #     "values": [32, 64, 128, 256, 512]
#         # },
#         # "n_layer_2": {
#         #     # Choose from pre-defined values
#         #     "values": [32, 64, 128, 256, 512, 1024]
#         # },
#         "batch_size":{
#             "distribution": "int_uniform", 
#             "min": 64, 
#             "max": 320,
#         },

#         "epochs":{
#             "distriubtion": "int_uniform",
#             "min": 80, 
#             "max": 120, 
#         },

#         "lr": {
#             # log uniform distribution between exp(min) and exp(max)
#             "distribution": "log_uniform",
#             "min": -9.21,   # exp(-9.21) = 1e-4
#             "max": -4.61    # exp(-4.61) = 1e-2
#         }

#     }
# }
hyperparameter_default={
    "batch_size": args.batch_size, 
    "lr": args.lr,
    "epochs":args.epochs,
    "optimizer_type":args.optimizer_type,
    "lr_scheduler": args.lr_scheduler,
}


class Dataset_Trainer:
    def __init__(self,
        # dataloader_type: str,
        # data_dir: str,
        # dataset_name: str,
        # data_mean: list, 
        # data_std: list, 
        # num_classes: int, 
        # img_size: int,
        # max_epochs: int, 
        # lr: float,
        # weight_decay: float,
        # scheduler: str,
        # lr_decay_steps: list[int],
        # batch_size: int,
        # num_workers: int,
        # gpus: list[int],
        # RandAug: list,
        hyperps,
    ):
        ## Dataloader arguments    
        self.dataset_name = args.dataset_name
        self.data_length = args.dataset_length
        self.num_classes=args.num_classes
        self.data_mean=args.dataset_mean
        self.data_std= args.dataset_std
        self.data_dir= args.data_dir
        self.img_size= args.img_size
        self.batch_size = hyperps["batch_size"]#args.batch_size
        self.num_workers = args.num_workers
        self.dataloader_type=args.dataloader_type

        ## Training Hyperparameters Arugments
        self.arch=args.arch
        self.arch_width_mu=args.width_mult
        self.gpus = args.gpus
        self.max_epochs=hyperps["epochs"] #args.epochs

        #self.optimizer_type=args.optimizer_type
        self.momentum= args.momentum
        self.m_transfer_enable=args.m_transfer_enable
        if args.arch=="resnet18":
            self.load_base_shapes=args.load_base_shapes
        elif args.arch=="resnet34": 
            print("loading res34 shape")
            self.load_base_shapes='/data/efficient_training/resnet34_shape.bsh'
        elif args.arch=="resnet50": 
            print("loading res50 shape")
            self.load_base_shapes='/data/efficient_training/resnet50_shape.bsh'
     
        else: 
            raise ValueError("Current Un-Support the architecture load_base_shape")

        self.lr = hyperps["lr"]#args.lr
        self.weight_decay = args.weight_decay   
        self.lr_decay_steps=args.steps
        self.scheduler = hyperps["lr_scheduler"]#args.scheduler
        self.optimizer_type=hyperps["optimizer_type"]#args.optimizer_type
        self.metric=args.metric
        self.RandAug= args.RandAug
 

        self.dataloader = DataModule_lightning(
            data_dir=self.data_dir,
            dataset_name = self.dataset_name,
            img_size = self.img_size,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            dataloader_type= self.dataloader_type,
            RandAug= self.RandAug,

        )

        self.pl_model = lightning_module(
            backbone_architecture=self.arch,
            arch_width_mu=self.arch_width_mu,
            load_base_shapes=self.load_base_shapes,
            m_transfer_enable=self.m_transfer_enable,
            num_classes=self.num_classes,
            batch_size = self.batch_size,
            data_length=self.data_length,
            epochs=self.max_epochs, 

            lr = self.lr,
            optimizer_type=self.optimizer_type,
            momentum=self.momentum,
            weight_decay = self.weight_decay,
            scheduler = self.scheduler,
            lr_decay_steps = self.lr_decay_steps,
            metric=self.metric,
            init_type=args.init_type,
        
        )

    
        self.wandb_logger = WandbLogger(
            name = f'{args.method}{self.dataset_name}{self.arch}{args.width_mult}{self.optimizer_type} lr={self.lr}{self.scheduler}batch_size{self.batch_size}',
            project = 'training_efficient',
            entity = 'mlbrl',
            group = self.dataset_name,
            job_type = args.job_type,
            offline = False,
        )
        callbacks_list=[]
        self.wandb_logger.watch(self.pl_model, log="gradients",  log_freq = 50)
        self.wandb_logger.log_hyperparams(args)
        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks_list.append(lr_monitor)
        

        self.trainer = Trainer(
            #fast_dev_run=True,
            accelerator = 'gpu',
            gpus = self.gpus,
            logger = self.wandb_logger,
            #max_steps = self.max_steps,
            max_epochs=self.max_epochs,
            strategy = 'ddp',
            callbacks= callbacks_list,
            #replace_sampler_ddp=True,
        )

    def run(self):
        seed_everything(10)
        print(f"Start Training : {self.dataset_name}")
        # for x, y in self.dataloader.train_dataloader(): 
        #     print(x.shape)
        self.trainer.fit(self.pl_model, self.dataloader)
        print("End Training")


    def coord_check_loader(self):
        transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((self.img_size, self.img_size), InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        dataloader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False)  
        
        return dataloader

    def coord_check_pl(self, mup, nsteps,nseeds,plotdir, legend=True, device='cuda'): 
        '''
        args: 
            mup
            nsteps: 
            nseeds: 
            plotdir
            legend
        '''
    
        optimizer=args.optimizer_type.replace('Mu', '')

        def gen(w, standparam=False): 
            def f(): 
                # model=getattr(resnet, args.arch)(num_classes=args.num_classes, wm=w).to(device)
                # base_model=getattr(resnet, args.arch)(num_classes=args.num_classes, wm=8).to(device)

                model= ResNet50(num_classes=args.num_classes,wm=w).to(device)
                base_model=ResNet50(num_classes=100, wm=10).to(device)
                #model=self.pl_model.model
                if standparam: 
                    set_base_shapes(model, None)
                    for param in model.parameters():
                        ### If initializing manually with fixed std or bounds,
                        ### then replace with same function from mup.init
                        #torch.nn.init.kaiming_normal(param,a=0)
                        torch.nn.init.uniform_(param, -0.1, 0.1)
                else: 
                    set_base_shapes(model, self.load_base_shapes)
                    #set_base_shapes(model, base_model)
                
                    for param in model.parameters():
                        ### If initializing manually with fixed std or bounds,
                        ### then replace with same function from mup.init
                        # torch.nn.init.uniform_(param, -0.1, 0.1)
                        init.uniform_(param, -0.1, 0.1)
                        
                        #mup.init.uniform_(param, -0.1, 0.1)
                return model 
            return f

        widths = 2**np.arange(-2., 4)
        models = {w: gen(w, standparam=not mup) for w in widths}
        dataloader=self.coord_check_loader()
        df = get_coord_data(models, dataloader, mup=mup, lr=self.lr, optimizer=optimizer, nseeds=nseeds, nsteps=nsteps)

        prm = 'μP' if mup else 'SP'
        plot_coord_data(df, legend=legend,
            save_to=os.path.join(plotdir, f'{prm.lower()}_{args.arch}_{optimizer}_coord.png'),
            suptitle=f'{prm} {args.arch} {optimizer} lr={self.lr} nseeds={nseeds}',
            face_color='xkcd:light grey' if not mup else None)
        


run_experiment=Dataset_Trainer(hyperparameter_default)

if args.coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        run_experiment.coord_check_pl(mup=True,
        nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, plotdir=args.plotdir, legend=False)
        
        run_experiment.coord_check_pl(mup=False,
         nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, plotdir=args.plotdir, legend=False)
        import sys; sys.exit()

run_experiment.run()

# function=run_experiment.run()
# wandb.agent(sweep_id, function=function)


#if __name__=="__main__": 
    
#     # parser = argparse.ArgumentParser(description=''
#     # '''
#     # PyTorch-Lightning Training, with μP or Standard Pytorch.

#     # To save base shapes info, run e.g.
#     #     python run.py --save_base_shapes resnet18.bsh --width_mult 1

#     # To train using MuAdam (or MuSGD), run
#     #     python run.py --width_mult 2 --load_base_shapes resnet18.bsh --optimizer {muadam,musgd}

#     # To test coords, run
#     #     python run.py --load_base_shapes resnet18.bsh --optimizer sgd --lr 0.1 --coord_check

#     # If you don't specify a base shape file, then you are using standard parametrization, e.g.
#     #     python run.py --width_mult 2 --optimizer {muadam,musgd}

#     # Note that models of different depths need separate `.bsh` files.
#     # ''', formatter_class=argparse.RawTextHelpFormatter)

    # run_experiment.run()