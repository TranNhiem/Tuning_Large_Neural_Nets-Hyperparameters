import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim import SGD, LBFGS
from torchmetrics import Accuracy
from typing import Optional, Sequence
from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout
import os

class MLP(nn.Module):
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0
    init_std= 1.0
    ):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = MuReadout(width, num_classes, bias=False, output_mult=output_mult)
        self.init_std= init_std
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        self.fc_1.weight.data *= self.init_std
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        self.fc_2.weight.data *= self.init_std
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out)


class MLP_mtransfer(pl.LightningModule): 
    
    def __init__(
        self, 
        dataset_name: str,
        max_steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        scheduler: str,
        lr_decay_steps: Optional[Sequence[int]] = None,
        task: str = 'linear_eval',
        **kwargs,
    ):
        super().__init__()
     
        self.num_classes = 10
        self.max_steps = max_steps
        # self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.task = task
        self.scheduler = scheduler
        print(self.lr)
        self.__build_model()

        self.mean_acc = Accuracy(num_classes=self.num_classes, average='macro')
        self.accuracy_1= Accuracy()
        self.accuracy_5= Accuracy(top_k=5)
        print(self.metric)

    
    def __build_model(self):

    
        ## 3..Adding the linear model corresponding output of class
       
        self.classifier = nn.Linear(2048, self.num_classes)


    def forward(self, x):

    
        x  = self.classifier(x)

        return x


    def training_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        #print("this is batch size", y.shape)
        y_logits = self.forward(x)
        # 2. Compute loss
        train_loss = F.cross_entropy(y_logits, y)
        acc1 = self.accuracy_1(y_logits, y, )
        acc5 = self.accuracy_5(y_logits, y, )
        log = { "train_loss": train_loss, "train_acc1": acc1, "train_acc5": acc5}

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        batch_size = x.size(0)
        y_logits = self.forward(x)

        # 2. Compute loss
        val_loss = F.cross_entropy(y_logits, y)
        acc1 = self.accuracy_1(y_logits, y, )
        acc5 = self.accuracy_5(y_logits, y, )
        results = {"batch_size": batch_size, "val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}

        return results

    def validation_epoch_end(self, outs):

        val_loss = self.__weighted_mean(outs, "val_loss", "batch_size") 
        val_acc1 = self.__weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = self.__weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        # print("this is batch size", y.shape)
        batch_size = x.size(0)
        y_logits = self.forward(x)

        # 2. Compute loss
        test_loss = F.cross_entropy(y_logits, y)
        acc1 = self.accuracy_1(y_logits, y, )
        acc5 = self.accuracy_5(y_logits, y, )
        results = {"batch_size": batch_size, "test_loss": test_loss, "test_acc1": acc1, "test_acc5": acc5}

        return results

    def test_epoch_end(self, outs):
        test_loss = self.__weighted_mean(outs, "test_loss", "batch_size")
       
        test_acc1 = self.__weighted_mean(outs, "test_acc1", "batch_size")
        test_acc5 = self.__weighted_mean(outs, "test_acc5", "batch_size")
        log = {"test_loss": test_loss, "test_acc1": test_acc1, "test_acc5": test_acc5}

        self.log_dict(log, sync_dist=True)

    def configure_optimizers(self):

        if self.task == 'finetune':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        else:
            optimizer = SGD(self.classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        
        if self.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler == 'reduce':
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=20, factor=0.5,),  "monitor": "val_loss"}
            return [optimizer], [scheduler]
        else: 
            return optimizer
        
    def __weighted_mean(self, outputs, key, batch_size_key):
        value = 0
        n = 0
        for out in outputs:
            value += out[batch_size_key] * out[key]
            n += out[batch_size_key]
        value = value / n
        return value.squeeze(0)
        
