import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from  sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim


# Create a SummaryWriter instance
writer = SummaryWriter(log_dir="logs")


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output


def train_model(model, train_loader, epoch, num_epochs, optimizer, log_every=100):
    _ = model.train()
    if torch.cuda.is_available():
        model.cuda()
    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        prediction = model.forward(image.float())
        
        loss = F.binary_cross_entropy_with_logits(
            prediction[0], label[0], weight=weight[0])

        loss.backward()
        optimizer.step()
        
        y_pred = torch.sigmoid(prediction).item()
        y_true = int(label.item())

        y_preds.append(y_pred)
        y_trues.append(y_true)

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        loss_value = loss.item()
        losses.append(loss_value)
        writer.add_scalar('Train/Loss', loss_value,
                        epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                    | avg train loss {4} | train auc : {5}'''.
                format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(train_loader),
                    np.round(np.mean(losses), 4),
                    np.round(auc, 4)
                )
                )
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, validation_loader, epoch, num_epochs, optimizer, log_every=100):
    _ = model.eval()    
    if torch.cuda.is_available():
        model.cuda()
    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(validation_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        prediction = model(image.float())
        
        loss = F.binary_cross_entropy_with_logits(prediction[0], label[0], weight=weight[0])
        
        y_pred = torch.sigmoid(prediction).item()
        y_true = int(label.item())

        y_preds.append(y_pred)
        y_trues.append(y_true)

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        loss_value = loss.item()
        losses.append(loss_value)

        writer.add_scalar('Valid/Loss', loss_value,
                        epoch * len(validation_loader) + i)
        writer.add_scalar('Valid/AUC', auc, epoch * len(validation_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                    | avg train loss {4} | train auc : {5}'''.
                format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(validation_loader),
                    np.round(np.mean(losses), 4),
                    np.round(auc, 4)
                )
                )
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)
    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    return val_loss_epoch, val_auc_epoch

