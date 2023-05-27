# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.optim import Adam
# from torchvision.datasets import MNIST
# from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
# from torch.utils.data import DataLoader


# def MNIST_loaders(train_batch_size=8000, test_batch_size=3000):
#     transform = Compose([
#         ToTensor(),
#         Normalize((0.1307,), (0.3081,))])

#     train_loader = DataLoader(
#         MNIST('./data/', train=True,
#               download=True,
#               transform=transform),
#         batch_size=train_batch_size, shuffle=True)

#     test_loader = DataLoader(
#         MNIST('./data/', train=False,
#               download=True,
#               transform=transform),
#         batch_size=test_batch_size, shuffle=False)

#     return train_loader, test_loader


# def overlay_y_on_x(x, y):
#     """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
#     """
#     x_ = x.clone()
#     x_ = x_.squeeze()
#     x_[:,0,:10] *= 0.0
#     x_[range(x.shape[0]),0,y] = x.max()
#     x_ = x_.unsqueeze(1)
#     return x_


# class Net(torch.nn.Module):

#     def __init__(self, dims):
#         super().__init__()
#         self.layers = []
#         for d in range(len(dims) - 1):
#             self.layers += [Layer(dims[d],dims[d+1]).cuda()]

#     def predict(self, x):
#         goodness_per_label = torch.zeros((10,x.shape[0])).cuda()
#         for label in range(10):
#             h = overlay_y_on_x(x, label)
#             for_next = h
#             goodness = torch.zeros(x.shape[0]).cuda()
#             for layer in self.layers:
#                 for_next,h = layer(for_next)
#                 temp = h.squeeze()
#                 goodness += temp.pow(2).sum([1,2])
#             goodness_per_label[label] = goodness
        
#         return goodness_per_label.argmax(0)

#     def train(self, x_pos, x_neg):
#         h_pos, h_neg = x_pos, x_neg
#         for i, layer in enumerate(self.layers):
#             print('training layer', i, '...')
#             h_pos, h_neg = layer.train(h_pos, h_neg)


# class Layer(torch.nn.Module):
#     def __init__(self, input, output):
#         super(Layer,self).__init__()
#         self.relu = torch.nn.ReLU()
        

#         ############
#         self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size= 7)
#         self.conv2 = torch.nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size= 7)
#         self.conv3 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size= 7, padding=3)
#         self.conv4 = torch.nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size= 7, padding=3)
#         self.conv5 = torch.nn.Conv2d(in_channels = 4, out_channels = 1, kernel_size= 7, padding=3)

        
#         ############
        
#         self.opt = Adam(self.parameters(), lr=0.01)
#         self.threshold = 2.0
#         self.num_epochs = 500

#     def forward(self, x):
#         ans = self.conv1(x)
#         ans = self.conv2(ans)
#         ans = self.conv3(ans)
#         ans = self.conv4(ans)
#         ans = self.conv5(ans)
#         ans = self.relu(ans)
#         x_direction = ans / (ans.norm(2, [2,3], keepdim=True) + 1e-4)

#         return x_direction, ans

#     def train(self, x_pos, x_neg):
#         for i in tqdm(range(self.num_epochs)):

#             out_pos, g_pos = self.forward(x_pos)
#             out_neg, g_neg = self.forward(x_neg)

#             g_pos = g_pos.pow(2).mean([2,3]).squeeze()
#             g_neg = g_neg.pow(2).mean([2,3]).squeeze()



#             # The following loss pushes pos (neg) samples to
#             # values larger (smaller) than the self.threshold.
#             loss =  torch.log(1 + torch.exp(torch.cat([
#                 -g_pos + self.threshold,
#                 g_neg - self.threshold]))).mean()
#             self.opt.zero_grad()
#             # this backward just compute the derivative and hence
#             # is not considered backpropagation.
#             loss.backward()
#             self.opt.step()

#         return out_pos.detach(), out_neg.detach()

    
# def visualize_sample(data, name='', idx=0):
#     reshaped = data[idx].cpu().reshape(28, 28)
#     plt.figure(figsize = (4, 4))
#     plt.title(name)
#     plt.imshow(reshaped, cmap="gray")
#     plt.show()
    
    
# if __name__ == "__main__":
#     torch.manual_seed(1234)
#     train_loader, test_loader = MNIST_loaders()
    
#     sample_x, sample_y = next(iter(train_loader))
#     sample_x, sample_y = sample_x.cuda(), sample_y.cuda()
#     sample_pos = overlay_y_on_x(sample_x, sample_y)
#     rnd = torch.randperm(sample_x.size(0))
#     sample_neg = overlay_y_on_x(sample_x, sample_y[rnd])
#     for data, name in zip([sample_x, sample_pos, sample_neg], ['orig', 'pos', 'neg']):
#         visualize_sample(data, name)

#     net = Net([1,1,1])
#     train_ac = 0
#     train_total = 0
#     for x,y in train_loader:
#         x, y = x.cuda(), y.cuda()
#         x_pos = overlay_y_on_x(x, y)
#         rnd = torch.randperm(x.size(0))
#         x_neg = overlay_y_on_x(x, y[rnd])
    
#         net.train(x_pos, x_neg)
#         train_total += len(net.predict(x))
#         train_ac += net.predict(x).eq(y).float().sum().item()

#     print('train error:', 1- train_ac/train_total)

#     test_total = 0
#     test_ac = 0

#     for x_te, y_te in test_loader:
#         x_te, y_te = x_te.cuda(), y_te.cuda()
#         test_total += len(net.predict(x))
#         test_ac += net.predict(x).eq(y).float().sum().item()

    
#     print('test error:', 1- test_ac/test_total)








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
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

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
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__()
        self.relu = torch.nn.ReLU()

        #########################
        ##Delete this part if using fc layer alone
        self.conv1 = torch.nn.Conv1d(in_channels= 1, out_channels= 4, kernel_size= 7,padding = 3)
        self.conv2 = torch.nn.Conv1d(in_channels= 4, out_channels= 1, kernel_size= 7, padding = 3)
        #########################

        self.linear = torch.nn.Linear(in_features,out_features,bias = True)

        self.threshold = 1.0
        self.num_epochs = 1000
        self.opt = Adam(self.parameters(), lr=0.03)

    def forward(self, x):
        ##########################
        ## With conv layer

        x_direction = (x / (x.norm(2, 1, keepdim=True) + 1e-4)).squeeze(0).unsqueeze(1)
        out = self.conv1(x_direction)
        out = self.conv2(out)
        out = out.squeeze()
        out = self.linear(out)
        ##########################
        ##with fc alone

        # x_direction = (x / (x.norm(2, 1, keepdim=True) + 1e-4)).squeeze(0)
        # out = self.linear(x_direction)

        ##########################
        out = self.relu(out)

        return out

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)
    
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
