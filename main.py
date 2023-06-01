import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def MNIST_loaders(train_batch_size=500, test_batch_size=500):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_ = x_.squeeze()
    x_[:,0,:10] *= 0.0
    x_[range(x.shape[0]),0,y] = x.max()
    x_ = x_.unsqueeze(1)
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d],dims[d+1]).cuda()]

    def predict(self, x):
        goodness_per_label = torch.zeros((10,x.shape[0])).cuda()
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = torch.zeros(x.shape[0]).cuda()
            count = 1
            for layer in self.layers:
                _,h = layer(h)
                if count > 1:
                    goodness += h.pow(2).sum([1,2,3])
                count += 1
            goodness_per_label[label] = goodness
        
        return goodness_per_label.argmax(0)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(torch.nn.Module):
    def __init__(self, input, output):
        super(Layer,self).__init__()
        self.relu = torch.nn.ReLU()
        

        ############
        self.conv1 = torch.nn.Conv2d(in_channels = input, out_channels = 128, kernel_size= 7,padding = 3)
        self.conv2 = torch.nn.Conv2d(in_channels = 128, out_channels = output, kernel_size= 7, padding = 3)
        self.ln = torch.nn.LayerNorm(28)
        self.ln1 = torch.nn.LayerNorm(28)
        ############
        
        self.opt = Adam(self.parameters(), lr=0.001)
        self.threshold = 2.0
        self.num_epochs = 50

    def forward(self, x):
        x_direction = x / (x.norm(2, [2,3], keepdim=True) + 1e-4)
        ans = self.conv1(x_direction)
        ans = self.ln1(ans)
        ans = self.relu(ans)
        ans = self.conv2(ans)
        ans = self.ln(ans)
        ans = self.relu(ans)

        return ans, ans

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):

            out_pos, g_pos = self.forward(x_pos)
            out_neg, g_neg = self.forward(x_neg)

            g_pos = g_pos.pow(2).mean([2,3]).squeeze()
            g_neg = g_neg.pow(2).mean([2,3]).squeeze()



            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss =  torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()

        return out_pos.detach(), out_neg.detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
def main_():
    torch.manual_seed(1222)
    train_loader, test_loader = MNIST_loaders()
    
    sample_x, sample_y = next(iter(train_loader))
    sample_x, sample_y = sample_x.cuda(), sample_y.cuda()
    sample_pos = overlay_y_on_x(sample_x, sample_y)
    rnd = torch.randperm(sample_x.size(0))
    sample_neg = overlay_y_on_x(sample_x, sample_y[rnd])
    for data, name in zip([sample_x, sample_pos, sample_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)

    net = Net([1,1,1,1])

    train_errors = []
    count  = 0
    for x,y in train_loader:
        if count > 50:
            break

        x, y = x.cuda(), y.cuda()
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])
    
        net.train(x_pos, x_neg)

        pred = net.predict(x)
        epoch_len = len(pred)
        ac_len = pred.eq(y).float().sum().item()

        error = 1- ac_len/epoch_len
        
        train_errors.append(error)

        print('error each batch: ', error)
        count += 1
    
    fin_error = sum(train_errors[len(train_errors)-3:])/3

    print('final train error:', fin_error)

    x_axis = list(range(len(train_errors)))


    plt.plot(x_axis, train_errors)
    plt.xlabel('batch')
    plt.xticks(x_axis)
    plt.ylabel('training error')
    plt.title('training error vs batch')
    plt.show()


    test_total = 0
    test_ac = 0

    print("evaluating on test set...")
    for x_te, y_te in tqdm(test_loader):
        x_te, y_te = x_te.cuda(), y_te.cuda()
        pred = net.predict(x_te)
        test_total += len(pred)
        test_ac += pred.eq(y_te).float().sum().item()

    
    print('test error:', 1-test_ac/test_total)