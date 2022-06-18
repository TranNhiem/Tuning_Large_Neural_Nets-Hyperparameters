
## Tran Nhiem -- 2022/05 
from typing import Optional, Sequence 
from torch.optim import SGD, Adam, AdamW
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LinearLR, CosineAnnealingLR,CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import torch.nn.functional as F 
from torchmetrics import Accuracy 
from convNet_architectures.ResNet import resnet18, resnet50
from convNet_architectures.resnet import ResNet18, ResNet34, ResNet50
## For Mup Optimizer 
from mup import MuAdam, MuSGD,MuAdamW 
## Coordinate Check
from mup.init import (kaiming_normal_, kaiming_uniform_, normal_,
                         trunc_normal_, uniform_, xavier_normal_,
                         xavier_uniform_)
from mup import get_shapes, make_base_shapes, set_base_shapes

init_type={
    "uniform": uniform_,
    "kaiming_normal": kaiming_normal_,
    "kaiming_uniform": kaiming_uniform_,
    "normal": normal_,
    "xavier_noraml":xavier_normal_,
    "xavier_uniform":xavier_uniform_,
}
class lightning_module(pl.LightningModule): 
    def __init__(self, 
        backbone_architecture: str, 
        batch_size: int ,  
        lr: float, 
        scheduler: str,
        weight_decay: float, 
        num_classes: int , 
        arch_width_mu: int or float,
        m_transfer_enable: bool,
        optimizer_type: str,
        init_type: str, 
        load_base_shapes: str,
        momentum: float,
        epochs: int, 
        data_length: int, 
        lr_decay_steps: Optional[Sequence[int]] = None,
        task: Optional[str]="classification",
        metric: str = "Accuracy", 
 
        **kwargs
    ):
        super().__init__()
        self.backbone_architecture= backbone_architecture
        self.width_mult=arch_width_mu
        self.m_transfer=m_transfer_enable
        self.load_base_shapes=load_base_shapes
        self.batch_size= batch_size 
        self.optimizer_type= optimizer_type
        self.momentum=momentum
        self.num_classes= num_classes
        self.lr = lr
        self.epochs=epochs
        self.data_length=data_length
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.task = task
        self.scheduler = scheduler
        self.init_type=init_type
        print(self.lr)
        self.__build_model()
        self.accuracy_1= Accuracy()
        self.accuracy_5= Accuracy(top_k=5)

        self.metric=metric
        self.criterion = nn.CrossEntropyLoss()

    def __build_model(self): 
        
        if self.backbone_architecture=="resnet18": 
            print('implement Res18 Mtransfer technique')
            self.model= ResNet18(num_classes=self.num_classes,wm=self.width_mult )
        elif self.backbone_architecture=="resnet34": 
            print('implement Res34 Mtransfer technique')
            self.model= ResNet34(num_classes=self.num_classes, wm=self.width_mult)
        elif self.backbone_architecture=="resnet50": 
            print('implement Res50 Mtransfer technique')
            self.model= ResNet50(num_classes=self.num_classes, wm=self.width_mult)
        elif self.backbone_architecture=="resnet18_v0": 
            print("Using Res18 Custome Build ResNet")
            self.model= resnet18(num_classes=self.num_classes,width_mult=self.width_mult,m_transfer=self.m_transfer  )
        elif self.backbone_architecture=="resnet50_v0": 
            print("Using Res50 Custome Build ResNet")
            self.model= resnet50(num_classes=self.num_classes,width_mult=self.width_mult,m_transfer=self.m_transfer)
        
        else: 
            raise ValueError("Current backbone is not support")
        
        if self.m_transfer: 

            if self.load_base_shapes: 
                print("implementation of μP technique")
                set_base_shapes(self.model, self.load_base_shapes)
            else:
                print("implementation of Standard Pytorch technique")
                set_base_shapes(self.model, None)

            for param in self.model.parameters():
                init_=init_type[self.init_type]
                ### If initializing manually with fixed std or bounds,
                ### then replace with same function from mup.init
                # torch.nn.init.uniform_(param, -0.1, 0.1)
                if self.init_type=="uniform":
                    init_(param, -0.1, 0.1)
                else: 
                    init_(param)
    def forward(self, x): 
        x=self.model(x)
        return x

    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_logits=self.forward(x)
        #y_logits=nn.Softmax(y_logits)
        # print(y.shape)
        # print(y_logits)

        #print('ground_truth', y[0])
        #print("prediction", y_logits[0])
        
        #print("prediction", y_logits[0].argmax(dim=-1))
        train_loss=F.cross_entropy(y_logits, y)
        #train_loss = F.nll_loss(y_logits, y)
        #train_loss=self.criterion(y_logits, y)
        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            log = {"train_loss": train_loss, "train_acc1": acc1, "train_acc5": acc5}
        else:
            raise ValueError("The metric is not support yet")
        
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx): 
        x, y = batch
       
        batch_size=x.size(0)
        y_logits=self.forward(x)
        # print(y[1])
        # print(y_logits[1])

        #print('ground_truth', y[0])
        #print("prediction", y_logits[0])
        
        #print("prediction", y_logits[0].argmax(dim=-1))
        #val_loss=self.criterion(y_logits, y)
        #val_loss = F.nll_loss(y_logits, y)

        val_loss=F.cross_entropy(y_logits, y)
        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            #result = {"batch_size": batch_size,"val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}
            result = {"val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}

        else:
            raise ValueError("The metric is not support yet")
        self.log_dict(result,on_epoch=True, sync_dist=True)
        
        return val_loss

    # def validation_epoch_end(self, outs): 
    #     val_loss= self.__weighted_mean(outs, "val_loss","batch_size" )
    #     if self.metric=="Accuracy": 
    #         val_acc_1= self.__weighted_mean(outs, "val_acc1","batch_size")
    #         val_acc_5= self.__weighted_mean(outs, "val_acc5", "batch_size")
    #         logs={"val_loss": val_loss, "val_acc_1": val_acc_1, "val_acc_5":val_acc_5}
        
    #     else: 
    #         raise ValueError("The metric is not support yet")
        
    #     self.log_dict(logs, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx): 
        x, y = batch
        batch_size=x.size(0)
        y_logits=self.forward(x)
        test_loss=F.cross_entropy(y_logits, y)
        #train_loss = F.nll_loss(y_logits, y)

        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            result = {"test_loss": test_loss, "test_acc1": acc1, "test_acc5": acc5}
            
            #result = {"batch_size": batch_size,"test_loss": val_loss, "test_acc1": acc1, "test_acc5": acc5}
        else:
            raise ValueError("The metric is not support yet")
        
        self.log_dict(result,on_epoch=True, sync_dist=True)
        
        return test_loss
    # def test_epoch_end(self, outs):
    #     test_loss= self.__weighted_mean(outs, "val_loss","batch_size" )
    #     if self.metric=="Accuracy": 
    #         test_acc_1= self.__weighted_mean(outs, "test_acc1","batch_size")
    #         test_acc_5= self.__weighted_mean(outs, "test_acc5", "batch_size")
    #         logs={"test_loss": test_loss, "test_acc_1": test_acc_1, "test_acc_5":test_acc_5}
        
    #     else: 
    #         raise ValueError("The metric is not support yet")
        
    #     self.log_dict(logs, on_epoch=True, sync_dist=True)
    

    def configure_optimizers(self):

        if self.optimizer_type =="MuSGD":
            optimizer = MuSGD(self.parameters(), lr=self.lr,momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_type =="MuAdam":
            optimizer=MuAdam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type=="MuAdamW": 
            optimizer=MuAdam(self.parameters(), lr=self.lr)
        ## Configure Type of Optimizer 
        elif self.optimizer_type =="SGD": 
            optimizer= SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay = self.weight_decay, nesterov=True)
        elif self.optimizer_type =="Adam":
            optimizer= Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay= self.weight_decay)
        elif self.optimizer_type=="AdamW": 
            optimizer= AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay= self.weight_decay)
        else: 
            raise ValueError('invalid Optimzier_type') 
    

       ## Configure Learning Rate Schedule
        if self.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler == 'reduce_plateau':
            #scheduler = ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=8, factor=0.5,),  "monitor": "val_loss"}
            return [optimizer], [scheduler]
        elif self.scheduler=="linear": 
            scheduler = LinearLR( optimizer, start_factor=0.5, total_iters=self.epochs/2)
            return [optimizer], [scheduler]
        elif self.scheduler=="cosineAnnealing": 
            scheduler = CosineAnnealingLR(optimizer, eta_min=1e-8,T_max=int((self.data_length/self.batch_size)*self.epochs) )
            return [optimizer], [scheduler]
        elif self.scheduler=="ConsineAnl_warmup": 
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=int(((self.data_length/self.batch_size)*self.epochs)/2), T_mult=1, eta_min=1e-8)
            return [optimizer], [scheduler]
        else: 
            print('you are not implementing any Learning schedule')
            return optimizer


    def __weighted_mean(self, outputs, key, batch_size_key):
        value=0
        n=0
        for out in outputs: 
            value += out[batch_size_key] +out[key]
            n+= out[batch_size_key]
        value=value/n 
        return value.squeeze(0)

if __name__=="__main__": 
    print("Hoooray ~ You are testing the lightning Module ~")
    print("Oh wait you are not define any testing module yet")
