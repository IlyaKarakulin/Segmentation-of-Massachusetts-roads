import torch
import torch.nn as nn
import torch.nn.functional as F

class gen_loss(nn.Module):
  def __init__(self,gamma=1.5,batch=True):
    super(gen_loss,self).__init__()
    self.bce_loss=nn.BCELoss()
    self.gamma=gamma

  def gen_dice(self,y_pred,y_true):
    epsilon=1e-8
    l1=abs(y_pred-y_true)**self.gamma
    y_pred_sqsum=torch.sum((y_pred*y_pred))
    y_true_sqsum=torch.sum((y_true*y_true))
    l1_sum=torch.sum(l1)
    score=(l1_sum + epsilon)/(y_pred_sqsum + y_true_sqsum )
    return score.mean()

  def __call__(self,y_pred,y_true):
    a=self.bce_loss(y_pred,y_true)
    b=self.gen_dice(y_pred,y_true)
    return a+b
  

def y_bce_loss(prediction1,prediction2,prediction3,prediction4,prediction5,label):
    dice=gen_loss()
    loss1=dice(prediction1,label)
    label=torch.nn.functional.interpolate(label, size=(256,256), scale_factor=None, mode='nearest')
    loss2=dice(prediction2,label)
    label=torch.nn.functional.interpolate(label, size=(128,128), scale_factor=None, mode='nearest')
    loss3=dice(prediction3,label)
    label=torch.nn.functional.interpolate(label, size=(64,64), scale_factor=None, mode='nearest')
    loss4=dice(prediction4,label)
    label=torch.nn.functional.interpolate(label, size=(32,32), scale_factor=None, mode='nearest')
    loss5=dice(prediction5,label)
    loss=loss1+loss2+loss3+loss4+loss5
    return loss

