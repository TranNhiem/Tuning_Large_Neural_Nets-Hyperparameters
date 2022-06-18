from py_lightning_modules.py_lightning_dataModule import DataModule_lightning
from py_lightning_modules.py_lightning_models import  lightning_module
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
import argparse 
from mup.coord_check import get_coord_data, plot_coord_data
import wandb

##**************************************************************************
## Setting Some of Hyperparameters 
##**************************************************************************

parser= argparse.ArgumentParser()
## Dataset Define
parser.add_argument('--dataloader_type', type=str, default='others', choices=['ffcv_loader', 'others'])
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
parser.add_argument('--arch', type=str, default='resnet18_v0', choices=['resnet18', 'resnet50'])
parser.add_argument('--width_mult', type=int, default=1)
parser.add_argument('--m_transfer_enable', type=bool, default=False)
parser.add_argument('--metric', type=str, default='Accuracy')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optimizer_type', default='SGD', )#choices=['SGD', 'Adam','AdamW' 'MuSGD', 'MuAdam', 'MuAdamW'])
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
parser.add_argument('--load_base_shapes', type=str, default='',
                    help='file location to load base shapes from')

## Visualization and Debug Setting
parser.add_argument("--method", type=str, default="py_standard")
parser.add_argument("--job_type", type=str, default="hyperparameter tuning")

parser.add_argument('--coord_check', action='store_true',
                    help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
parser.add_argument('--coord_check_nsteps', type=int, default=3,
                    help='Do coord check with this many steps.')
parser.add_argument('--coord_check_nseeds', type=int, default=1,
                    help='number of seeds for coord check')

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

##*****************************************************************
## If Running Sweeping for Searching Hyperprameters
##*****************************************************************

# hyperparameter_default=dict(
#     batch_size=128, 
#     lr=1e-3,
#     epochs=90,
#     optimizer_type="SGD",
#     lr_scheduler="ConsineAnl_warmup",
# )

# wandb.init(config=hyperparameter_default, 
#             #name = f'{args.method} {self.dataset_name} arch={self.arch} optimizer={self.optimizer_type} lr={self.lr} lr_schedule={self.scheduler} wd={self.weight_decay} batch_size {self.batch_size}',
#             name = f'{args.method} {args.dataset_name} arch={args.arch}',
#             project = 'training_efficient',
#             entity = 'mlbrl',
#             group = args.dataset_name,
#             job_type = args.job_type,
            
#             )
# config=wandb.config

hyperparameter_default={
    "batch_size": args.batch_size, 
    "lr": args.lr,
    "epochs":args.epochs,
    "optimizer_type": args.optimizer_type,
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
        hyperps: dict,
        **kwargs
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
        self.load_base_shapes=args.load_base_shapes
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
            name = f'{args.method} {self.dataset_name} arch={self.arch} optimizer={self.optimizer_type} lr={self.lr} lr_schedule={self.scheduler} wd={self.weight_decay} batch_size {self.batch_size}',
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


  

    # def coord_check(self, mup, lr, optimizer, nsteps, base_shape, 
    #                 nseeds, device='cuda', plotdir, legend='True'): 
    #     optimizer=optimizer.replace('Mu', '')

    #     def gen(w, standparam=False): 
    #         def f(): 
    #             model=self.pl_model.model
    #             if standardparam: 
    #                 set_base_shapes(model, None)
    #             else: 
    #                 set_base_shapes(model, base_shape)
    


run_experiment=Dataset_Trainer(hyperparameter_default)
run_experiment.run()
# function=run_experiment.run()
# wandb.agent(sweep_id, function=function)


# if __name__=="__main__": 
    
# #     # parser = argparse.ArgumentParser(description=''
# #     # '''
# #     # PyTorch-Lightning Training, with μP or Standard Pytorch.

# #     # To save base shapes info, run e.g.
# #     #     python run.py --save_base_shapes resnet18.bsh --width_mult 1

# #     # To train using MuAdam (or MuSGD), run
# #     #     python run.py --width_mult 2 --load_base_shapes resnet18.bsh --optimizer {muadam,musgd}

# #     # To test coords, run
# #     #     python run.py --load_base_shapes resnet18.bsh --optimizer sgd --lr 0.1 --coord_check

# #     # If you don't specify a base shape file, then you are using standard parametrization, e.g.
# #     #     python run.py --width_mult 2 --optimizer {muadam,musgd}

# #     # Note that models of different depths need separate `.bsh` files.
# #     # ''', formatter_class=argparse.RawTextHelpFormatter)

#     run_experiment.run()