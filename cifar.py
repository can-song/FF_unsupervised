import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import prepare_data
from sklearn.metrics import accuracy_score
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
# os.system('rm -rf ./logs')
writer = SummaryWriter(f'./logs/{datetime.now()}')

def get_writer()->SummaryWriter:
    return writer

def loss_func(x1, x2):

    pho = 0.1
    f = lambda x: -pho * torch.log(x.mean(dim=1)+1e-12) - (1-pho) * torch.log(1-x.mean(dim=1)+1e-12)
    # return torch.mean(f(x1) + f(x2) + 1 / ((x1-x2).pow(2).sum(dim=1, keepdim=True)+1e-6) )
    alpha = 0.01
    return torch.mean(alpha*f(x1) + alpha*f(x2) + torch.einsum("nihw,njhw->nhw", x1, x2)/(torch.norm(x1, dim=1)*torch.norm(x2, dim=1)+1e-5) )

def sparsity_loss(x1, x2, rho=0.1):
    # x1 = x1 * 0.5 + 0.5
    # x2 = x2 * 0.5 + 0.5
    # x1 = torch.mean(x1.pow(2), dim=1)
    f = lambda x: -rho * torch.log(x.mean(dim=0)+1e-6) - (1-rho) * torch.log(1-x.mean(dim=0)+1e-6)
    return (torch.mean(f(x1) + f(x2)) + rho * np.log(rho) + (1-rho) * np.log(1-rho))

def push_loss(x1, x2):
    # return -torch.mean((x1-x2).pow(2))
    # return torch.mean(1 / ((x1-x2).pow(2).sum(dim=1, keepdim=True)+1e-6))
    # return torch.mean(torch.einsum("nkhw,nkhw->nhw", x1, x2)/(torch.norm(x1, dim=1)*torch.norm(x2, dim=1)+1e-5))
    return torch.mean(torch.sum(x1, dim=1)+torch.sum(x2, dim=1))

def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)


class Layer(nn.Module):
    count = 0
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1):
        super().__init__()
        self.name = f"layer-{Layer.count:02d}"
        Layer.count += 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        # self.act = nn.Sigmoid()
        self.act = nn.Softplus()
        self.opt = torch.optim.Adam(self.parameters())
        
        self.loss = loss_func
        self.step = 0

    def layer_train(self, x1, x2):
        """
        Train the layer using positive and negative activations.
        """

        self.opt.zero_grad()
        # loss = self.loss(x1, x2)
        loss1 = sparsity_loss(x1, x2)
        loss2 = push_loss(x1, x2)
        loss = loss1 + loss2
        get_writer().add_scalar(f"{self.name}/sparsity", loss1, self.step)
        get_writer().add_scalar(f"{self.name}/push", loss2, self.step)
        self.step += 1
        loss.backward()
        self.opt.step()

    def forward(self, x):
        x = self.conv(x.detach())
        x = self.act(x)
        x = x / (torch.norm(x, dim=1, keepdim=True)+1e-5)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_epochs = 5
        self.classifier_epochs = 1
        self.feature_batch_size = 64
        self.classifier_batch_size = 64

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', download=False, transform=transform, train=True)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', download=False, transform=transform, train=False)
        
        self.layers = nn.ModuleList([
            Layer(3, 64, 5, 2),
            # Layer(64, 64, 3, 1),
            nn.MaxPool2d(3, 2, 1),
            Layer(64, 128, 5, 2),
            nn.MaxPool2d(3, 2, 1),
            Layer(128, 256, 5, 2),
            nn.MaxPool2d(3, 2, 1),
            Layer(256, 1024, 4, 0),
            nn.Flatten()
        ])

        self.linear_layer = nn.Sequential(
            nn.Linear(1024, 10, bias=False)
        )
        self.opt = torch.optim.Adam(self.linear_layer.parameters())
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def train_layers(self):
        outer_tqdm = tqdm(range(self.feature_epochs), desc="Training Layers", position=0)
        for epoch in outer_tqdm:
            dataloader1 = DataLoader(self.train_dataset, batch_size=self.feature_batch_size, shuffle=True, num_workers=4)
            dataloader2 = DataLoader(self.train_dataset, batch_size=self.feature_batch_size, shuffle=True, num_workers=4)

            inner_tqdm = tqdm(zip(dataloader1, dataloader2), desc=f"Training Layers | Epoch {epoch}",
                              leave=False, position=1)
            for x1, x2 in inner_tqdm:
                x1, x2 = x1[0].to(self.device), x2[0].to(self.device)

                for idx, layer in enumerate(self.layers):
                    
                    x1, x2 = layer(x1), layer(x2)
                    
                    if hasattr(layer, "layer_train"):
                        layer.layer_train(x1, x2)
                # break
            if epoch % 100 == 0:
                    torch.save(self.state_dict(), f"mnist_{epoch}.pt")

    def train_classifier(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.classifier_batch_size, shuffle=True, num_workers=4)
        num_examples = len(dataloader)
        outer_tqdm = tqdm(range(self.classifier_epochs), desc="Training Classifier", position=0)
        loss_list = []
        step = 0
        writer = get_writer()
        for epoch in outer_tqdm:
            epoch_loss = 0
            inner_tqdm = tqdm(dataloader, desc=f"Training Last Layer | Epoch {epoch}", leave=False, position=1)
            for x, y in inner_tqdm:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                x = self(x)
                loss = self.loss(x, y)
                epoch_loss += loss
                loss.backward()
                self.opt.step()
                writer.add_scalar('classifier loss', loss, step)
                step += 1
        #     loss_list.append(epoch_loss / num_examples)
        #     # Update progress bar with current loss
        # return [l.detach().cpu().numpy() for l in loss_list]

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        x = self.feature_forward(x)

        x = self.linear_layer(x)
        return x

    def feature_forward(self, x: torch.Tensor):
        x = x.to(self.device)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    def evaluate_linear(self, dataset_type: str = "test"):
        if dataset_type == "train":
            dataloader = DataLoader(self.train_dataset, batch_size=self.classifier_batch_size, shuffle=False, num_workers=4)
        else:
            dataloader = DataLoader(self.test_dataset, batch_size=self.classifier_batch_size, shuffle=False, num_workers=4)
        self.eval()
        inner_tqdm = tqdm(dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self(images)
            preds = torch.argmax(preds, 1)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
        all_labels = torch.concat(all_labels, 0).numpy()
        all_preds = torch.concat(all_preds, 0).numpy()
        metrics_dict = get_metrics(all_preds, all_labels)
        print(f"\n{dataset_type} dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))

    def evaluate_cluster(self):
        self.eval()
        
        
        batch_size = len(self.train_dataset)
        batch_size = 10000
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.classifier_batch_size, shuffle=False, num_workers=4)

        inner_tqdm = tqdm(train_dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self.feature_forward(images)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            break
        all_labels = torch.concat(all_labels, 0)
        all_preds = torch.concat(all_preds, 0)
        all_centers = []
        for idx in range(10):
            mask = torch.nonzero(all_labels == idx).squeeze()
            center = all_preds[mask, ...]
            # 0.7468
            center = center[random.sample(range(center.shape[0]), 10)]
            center = center.mean(0, keepdim=True)
            all_centers.append(center)
        all_centers = torch.stack(all_centers, dim=1)
        
        
        inner_tqdm = tqdm(test_dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self.feature_forward(images)
            
            # preds = torch.argmin(torch.log((preds - all_centers).pow(2)+1e-12).sum(dim=2), 1)
            preds = torch.argmin((preds[:, None] - all_centers).pow(2).sum(dim=2), 1)
            # preds = torch.argmax(preds, 1)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
        all_labels = torch.concat(all_labels, 0).numpy()
        all_preds = torch.concat(all_preds, 0).numpy()
        metrics_dict = get_metrics(all_preds, all_labels)
        print(f"\ntest dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))


def plot_loss(loss):
    # plot the loss over epochs
    fig = plt.figure()
    plt.plot(list(range(len(loss))), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.savefig("Loss Plot.png")
    # plt.show()


if __name__ == '__main__':
    net = Net()
    net.train_layers()
    net.evaluate_cluster()
    net.train_classifier()
    net.evaluate_linear("train")
    net.evaluate_linear("test")
    
    # # prepare_data()

    # # Load the MNIST dataset
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor()
    # ])
    # pos_dataset = torchvision.datasets.MNIST(root='./', download=False, transform=transform, train=True)
    # # pos_dataset = Subset(pos_dataset, list(range(1000)))
    # # Create the data loader
    # batch_size = 600
    # pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # Load the transformed images
    # # neg_dataset = torch.load('transformed_dataset.pt')
    # # Create the data loader
    # neg_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # Load the test images
    # test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)
    # # Create the data loader
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # unsupervised_ff = Unsupervised_FF(device=device, n_epochs=20)
    
    # # unsupervised_ff.load_state_dict(torch.load("mnist_1900.pt"))


    # loss = train(unsupervised_ff, pos_dataloader, neg_dataloader)
    # plot_loss(loss)

    # unsupervised_ff.evaluate(pos_dataloader, dataset_type="Train")
    # unsupervised_ff.evaluate(test_dataloader, dataset_type="Test")
    # unsupervised_ff.evaluate2()
