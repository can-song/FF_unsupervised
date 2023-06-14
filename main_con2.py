import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import prepare_data
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

import os
os.system('rm -rf ./logs')
writer = SummaryWriter('./logs')

def goodness_score(pos_acts, neg_acts, threshold=2):
    """
    Compute the goodness score for a given set of positive and negative activations.

    Parameters:

    pos_acts (torch.Tensor): Numpy array of positive activations.
    neg_acts (torch.Tensor): Numpy array of negative activations.
    threshold (int, optional): Threshold value used to compute the score. Default is 2.

    Returns:

    goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
    score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
    quantity but without the threshold subtraction
    """

    # pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    # neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    # return torch.add(pos_goodness, neg_goodness)
    # return -torch.log(torch.pow(pos_acts-neg_acts, 2)+1e-3).sum()
    # return -torch.sum(pos_acts * (torch.log(pos_acts+1e-3) - torch.log(neg_acts+1e-3)))
    # return - torch.sum(torch.exp(pos_acts)*(pos_acts - neg_acts)) \
    #        - torch.sum(torch.exp(neg_acts)*(neg_acts - pos_acts))
    
    # return -torch.sum(torch.exp(-pos_acts)*(neg_acts-pos_acts)) \
        #    -torch.sum(torch.exp(-neg_acts)*(pos_acts-neg_acts))
    # pos_acts = torch.log(torch.exp(pos_acts)-1+1e-3)
    # neg_acts = torch.log(torch.exp(neg_acts)-1+1e-3)
    # return torch.sum(pos_acts.pow(2) + neg_acts.pow(2) - (pos_acts-neg_acts).abs() * 2)
    # return torch.sum(0.5*pos_acts.pow(2) + 0.5*neg_acts.pow(2) - torch.log(torch.pow(pos_acts-neg_acts, 2)+1e-12))
    # return torch.sum(0.5*(pos_acts-1).pow(2) + 0.5*(neg_acts-1).pow(2) - torch.log(torch.pow(pos_acts-neg_acts, 2)+1e-12))
    # *return torch.sum(0.5*F.relu(pos_acts-0.5) + 0.5*F.relu(neg_acts-0.5) - torch.log(torch.pow(pos_acts-neg_acts, 2)+1e-12))
    # return pos_acts.pow(2).sum() + neg_acts.pow(2).sum() - torch.log(torch.pow(pos_acts-neg_acts, 2).sum())
    # return pos_acts.pow(2).mean() + neg_acts.pow(2).mean() - torch.log(torch.pow(pos_acts-neg_acts, 2).mean())
    # return pos_acts.abs().mean() + neg_acts.abs().mean() - torch.log(torch.pow(pos_acts-neg_acts, 2).mean())
    # f = lambda x: x.abs() - torch.log(torch.pow(x, 2)+1e-12) - 1
    # return torch.sum(f(pos_acts-neg_acts))
    # return torch.sum(f(pos_acts) + f(neg_acts) + f(pos_acts-neg_acts))
    # g = lambda x: torch.log((x-x[:, torch.randperm(x.shape[1])]).pow(2) + 1e-12)
    # return torch.sum(f(pos_acts) + f(neg_acts))# - torch.log((pos_acts-neg_acts).pow(2)+1e-12) - g(pos_acts) - g(neg_acts))
    # beta = 2 # 3
    # return torch.sum(pos_acts.pow(2) + neg_acts.pow(2) - beta*beta*torch.log((pos_acts-neg_acts).pow(2)+1e-12) - beta)
    # return torch.sum((pos_acts-neg_acts).pow(2) - torch.log((pos_acts-neg_acts).pow(2)+1e-12)) - 1
    
    return torch.sum(pos_acts.abs() + neg_acts.abs() - torch.log((pos_acts-neg_acts).pow(2)+1e-12))

def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, n_epochs: int, bias: bool, device):
        super().__init__(in_features, out_features, bias=bias)
        self.n_epochs = n_epochs
        self.goodness = goodness_score
        self.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)
        # self.act = nn.Sigmoid()
        # self.act = nn.Softmax(dim=1)
        # self.act = nn.LogSigmoid()
        # self.act = nn.ReLU()
        self.act = nn.Softplus()
        # self.act = nn.LeakyReLU(0.2)

        self.to(device)
        self.opt = torch.optim.Adam(self.parameters())

    def ff_train(self, pos_acts, neg_acts):
        """
        Train the layer using positive and negative activations.

        Parameters:

        pos_acts (numpy.ndarray): Numpy array of positive activations.
        neg_acts (numpy.ndarray): Numpy array of negative activations.
        """

        self.opt.zero_grad()
        goodness = self.goodness(pos_acts, neg_acts)
        goodness.backward()
        self.opt.step()

    def forward(self, input):
        # input = super().forward(input.detach())
        # input = self.ln_layer(input.detach())
        input = super().forward(input)
        # input = self.ln_layer(input)
        input = self.act(input)
        return input


class Unsupervised_FF(nn.Module):
    def __init__(self, n_layers: int = 4, n_neurons=500, input_size: int = 28 * 28, n_epochs: int = 100,
                 bias: bool = True, n_classes: int = 10, n_hid_to_log: int = 3, device=torch.device("cuda:0")):
        super().__init__()
        self.n_hid_to_log = n_hid_to_log
        self.n_epochs = n_epochs
        self.device = device

        # ff_layers = [
        #     FF_Layer(in_features=input_size if idx == 0 else n_neurons,
        #              out_features=n_neurons,
        #              n_epochs=n_epochs,
        #              bias=bias,
        #              device=device) for idx in range(n_layers)]
        ff_layers = nn.ModuleList([
            FF_Layer(28*28, 512, n_epochs, bias, device),
            FF_Layer(512, 256, n_epochs, bias, device),
            # FF_Layer(512, 512, n_epochs, bias, device),
            FF_Layer(256, 128, n_epochs, bias, device),
            # FF_Layer(128, 64, n_epochs, bias, device),
            # FF_Layer(64, 32, n_epochs, bias, device),
            # FF_Layer(32, 16, n_epochs, bias, device),
            # FF_Layer(16, 8, n_epochs, bias, device),
            # FF_Layer(8, 4, n_epochs, bias, device),
        ])
        
        optimizers = []
        models = []
        model = nn.Sequential()
        for idx, layer in enumerate(ff_layers):
            optimizer = torch.optim.Adam(layer.parameters())
            # optimizer.add_param_group({'params': layer.parameters(), 'lr':1e-3})
            optimizer.add_param_group({'params': model.parameters(), 'lr':1e-4})
            # print(optimizer.param_groups)
            model.append(layer)
            models.append(model)
            optimizers.append(optimizer)
            
        self.models = models
        self.optimizers = optimizers

        # 0.93115, 0.9271
        self.ff_layers = ff_layers
        self.last_layer = nn.Linear(in_features=128, out_features=n_classes, bias=bias)
        self.to(device)
        self.opt = torch.optim.Adam(self.last_layer.parameters())
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def train_ff_layers(self, pos_dataloader, neg_dataloader):
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training FF Layers", position=0)
        for epoch in outer_tqdm:
            batch_size = pos_dataloader.batch_size
            pos_dataset = torchvision.datasets.MNIST(root='./', download=False, transform=transform, train=True)
            pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            ff = False
            if ff:
                neg_dataset = torch.load('transformed_dataset.pt')
                # Create the data loader
                neg_dataloader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            else:
                neg_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            inner_tqdm = tqdm(zip(pos_dataloader, neg_dataloader), desc=f"Training FF Layers | Epoch {epoch}",
                              leave=False, position=1)
            for pos_data, neg_imgs in inner_tqdm:
                pos_imgs, _ = pos_data
                if ff:
                    neg_imgs = neg_imgs
                else:
                    neg_imgs, _ = neg_imgs
                pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(self.device)
                neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(self.device)
                # neg_acts = pos_acts[torch.randperm(pos_acts.shape[0])]

                for model, opt in zip(self.models, self.optimizers):
                    opt.zero_grad()
                    goodness = goodness_score(model(pos_acts), model(neg_acts))
                    goodness.backward()
                    opt.step()

                # for idx, layer in enumerate(self.ff_layers):
                #     pos_acts = pos_acts.detach()
                #     neg_acts = neg_acts.detach()
                #     pos_acts = layer(pos_acts)
                #     neg_acts = layer(neg_acts)
                #     layer.ff_train(pos_acts, neg_acts)

    def train_last_layer(self, dataloader: DataLoader):
        num_examples = len(dataloader)
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training Last Layer", position=0)
        loss_list = []
        for epoch in outer_tqdm:
            epoch_loss = 0
            inner_tqdm = tqdm(dataloader, desc=f"Training Last Layer | Epoch {epoch}", leave=False, position=1)
            for images, labels in inner_tqdm:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.opt.zero_grad()
                preds = self(images)
                loss = self.loss(preds, labels)
                epoch_loss += loss
                loss.backward()
                self.opt.step()
            writer.add_scalar("Loss/train", epoch_loss / num_examples, epoch)
            loss_list.append(epoch_loss / num_examples)
            # Update progress bar with current loss
        return [l.detach().cpu().numpy() for l in loss_list]

    def forward(self, image: torch.Tensor):
        image = image.to(self.device)
        image = torch.reshape(image, (image.shape[0], 1, -1))
        # concat_output = []
        for idx, layer in enumerate(self.ff_layers):
            image = layer(image)
            # if idx > len(self.ff_layers) - self.n_hid_to_log - 1:
                # concat_output.append(image)
        # concat_output = torch.concat(concat_output, 2)
        # logits = self.last_layer(concat_output)
        logits = self.last_layer(image)
        return logits.squeeze()

    def evaluate(self, dataloader: DataLoader, dataset_type: str = "train"):
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
        print(f"{dataset_type} dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))


def train(model: Unsupervised_FF, pos_dataloader: DataLoader, neg_dataloader: DataLoader):
    model.train()
    model.train_ff_layers(pos_dataloader, neg_dataloader)
    return model.train_last_layer(pos_dataloader)


def plot_loss(loss):
    # plot the loss over epochs
    fig = plt.figure()
    plt.plot(list(range(len(loss))), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.savefig("Loss Plot.png")
    plt.show()


if __name__ == '__main__':
    # prepare_data()

    # Load the MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    pos_dataset = torchvision.datasets.MNIST(root='./', download=False, transform=transform, train=True)
    # pos_dataset = Subset(pos_dataset, list(range(1000)))
    # Create the data loader
    batch_size = 64
    pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load the transformed images
    # neg_dataset = torch.load('transformed_dataset.pt')
    # Create the data loader
    neg_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load the test images
    test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)
    # Create the data loader
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    unsupervised_ff = Unsupervised_FF(device=device, n_epochs=20)

    loss = train(unsupervised_ff, pos_dataloader, neg_dataloader)

    # plot_loss(loss)

    unsupervised_ff.evaluate(pos_dataloader, dataset_type="Train")
    unsupervised_ff.evaluate(test_dataloader, dataset_type="Test")
