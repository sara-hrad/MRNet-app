import torch
import torch.nn as nn
import torchvision
import argparse
import datetime 
import os
from torchvision import transforms as t
from torch.utils.tensorboard import SummaryWriter
import sklearn.linear_model as sk
# import torch_geometric.transforms
from model import *
from data_loader import *
from util import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--task', type=str, required=True,
    #                     choices=['abnormal', 'acl', 'meniscus'], default='acl')
    # parser.add_argument('-p', '--plane', type=str, required=True,
    #                     choices=['sagittal', 'coronal', 'axial'], default='sagittal')
    parser.add_argument('-t', '--task', type=str, choices=['abnormal', 'acl', 'meniscus'], default='acl')
    parser.add_argument('-p', '--plane', type=str, choices=['sagittal', 'coronal', 'axial'], default='axial')
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, choices=[0, 1], default=5)

    args = parser.parse_args()
    return args

  
def run(args):
    log_root_folder = "/content/drive/MyDrive/MRNet-v1.0/logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)


def train():
    ##############################################################################
    ###################       Augmentation                  ######################
    ##############################################################################
    augmentor = t.Compose([t.Lambda(lambda x: torch.Tensor(x)),
                        t.RandomRotation(25),
                        t.RandomAffine(0,translate=[0.11, 0.11]),
                        t.RandomHorizontalFlip(),
                        t.RandomVerticalFlip(),
                        t.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
                        ])

    ##############################################################################
    ###################       Data for training             ######################
    ##############################################################################
    train_dataset = MRDataset('./data/MRNet-v1.0/', args.task, args.plane, transform=augmentor, train=True)
    validation_dataset = MRDataset('./data/MRNet-v1.0/', args.task, args.plane, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

    mrnet = MRNet()
    # mrnet = mrnet.cuda()

    optimizer = torch.optim.Adam(mrnet.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience

    for epoch in range(num_epochs):
        train_loss, train_auc = train_model(mrnet, train_loader, epoch, num_epochs, optimizer)
        val_loss, val_auc = evaluate_model(mrnet, validation_loader, epoch, num_epochs)

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3}".format(
            train_loss, train_auc, val_loss, val_auc))

        if args.lr_scheduler == 1:
            scheduler.step(val_loss)

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.task}_{args.plane}_val_auc_\{val_auc:0.4f}_train_auc_{train_auc:0.4f}\_epoch_{epoch+1}.pth'
                for f in os.listdir('./results/'):
                    if (args.task in f) and (args.plane in f):
                        os.remove(f'./results/{f}')
                torch.save(mrnet, f'./results/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                    format(iteration_change_loss))
            break


def regression():
    task = 'acl'
    results = {}
    results_val = {}

    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane)
        results['labels'] = labels
        results[plane] = predictions
        
    X = np.zeros((len(predictions), 3))
    X[:, 0] = results['axial']
    X[:, 1] = results['coronal']
    X[:, 2] = results['sagittal']

    y = np.array(labels)

    logreg = sk.LogisticRegression(solver='lbfgs')
    logreg.fit(X, y)

    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, train=False)
        results_val['labels'] = labels
        results_val[plane] = predictions

    X_val = np.zeros((len(predictions), 3))
    X_val[:, 0] = results_val['axial']
    X_val[:, 1] = results_val['coronal']
    X_val[:, 2] = results_val['sagittal']

    y_val = np.array(labels)

    y_pred = logreg.predict_proba(X_val)[:, 1]
    metrics.roc_auc_score(y_val, y_pred)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    for args.plane in ['sagittal', 'coronal', 'axial']:
        train()

    regression()
    