import torch
import torch.nn as nn
import torch.utils
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms

import argparse
from tensorboardX import SummaryWriter


BATCH_SIZE = 16 # how to choose batch size? https://www.zhihu.com/question/61607442/answer/440401209
EPOCH = 100
LEARNING_RATE = 0.001
REPORT_EVERY_BATCH = 500
PATH = "./cifar_net3.pth"


'''
nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
nn.functional.relu on the other side is just the functional API call to the relu function, 
so that you can add it e.g. in your forward method yourself. Generally speaking it might 
depend on your coding style if you prefer modules for the activations or the functional calls. 
'''

'''
consume that we use data from cifar-10, whose shape are all (32(H), 32(W), 3(channels))
there are 10 classes.
'''

'''
The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            # (BATCH_SIZE, C_in, H_in, W_in)
            # after conv2d, output is(if kernel is a square):
            # (BATCH_SIZE, C_out, ((H_in - kernel_size + 2*padding)/stride)+1, ((W_in - kernel_size + 2*padding)/stride)+1)
            # note as (BATCH_SIZE, c_out, H_out, W_out)

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # if input is (BATCH_SIZE, C_in, H_in, W_in), after maxpooling, output is(if kernel is a square):
            # (BATCH_SIZE, C_out, ((H_in - kernel_size + 2*padding)/stride)+1, ((W_in - kernel_size + 2*padding)/stride)+1)
            # the calculation method is the same as convolutional neural network.
            # most of the time, stride=kernel_size and padding=0, so output is (BATCH_SIZE, C_out, H_in/kernel_size, W_in/kernel_size)

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.dense = nn.Sequential(
            # we need to calculate the output shape of self.conv1 with initial input image shape=(32, 32, 3)
            # 1.after Conv2d: (BATCH_SIZE, 6, 28, 28)
            # 2.after ReLu: (BATCH_SIZE, 6, 28, 28)
            # 3.after MaxPool2d: (BATCH_SIZE, 6, 14, 14)
            # 4.after Conv2d: (BATCH_SIZE, 16, 10, 10)
            # 5.after ReLu: (BATCH_SIZE, 16, 10, 10)
            # 6.after MaxPool2d: (BATCH_SIZE, 16, 5, 5)

            # so the input of Linear should be 16*5*5 and it means that mathmetically we should give Linear a matrix
            # with size [BATCH_SIZE, 16*5*5], so it should be torch.size([BATCH_SIZE, 16*5*5])
            # pay attention: torch.tensor([1,2,3,4]) has torch.size([4])
            #                torch.tensor([[1,2,3,4]]) has torch.size([1, 4])

            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),

            # at last we can add an output layer using LogSoftmax(dim=1), it is the log of softmax.
            # directly use softmax may encounter overflow while using logsoftmax can avoid that.
            # softmax can rescale the values to [0, 1] and all values sum to 1. this matches the 
            # concept of probability. after doing softmax we can consider every value as the 
            # probility of belonging to one calss.
            # however sigmoid will just rescale values to [0, 1].
            
            # actually neural networks will consider the largest number as it's prediction
            # so when training a classifier the logsoftmax can be omitted. this will reduce computation.
            # and the result won't change!

            # The output from the NN is a probability distribution over actions, so a straightforward way to
            # proceed would be to include softmax nonlinearity after the last layer. However, in the preceding NN, 
            # we don't apply softmax to increase the numerical stability of the training process. Rather than 
            # calculating softmax (which uses exponentiation) and then calculating cross-entropy loss (which uses 
            # a logarithm of probabilities), we can use the PyTorch class nn.CrossEntropyLoss, which combines both 
            # softmax and cross-entropy in a single, more numerically stable expression. CrossEntropyLoss requires raw, 
            # unnormalized values from the NN (also called logits). The downside of this is that we need to remember 
            # to apply softmax every time we need to get probabilities from our NN's output.
            nn.LogSoftmax(dim=1)
            # nn.Softmax(dim=1) # failed!
            # nn.Sigmoid()      # failed!

            # now we can calculate the output shape of self.dense with initial input shape=(BATCH_SIZE, 16*5*5)
            # 1.after Linear: (BATCH_SIZE, 120)
            # 2.after ReLu(): (BATCH_SIZE, 120)
            # 3.after Dropout(): (BATCH_SIZE, 120)
            # 4.after Linear: (BATCH_SIZE, 84)
            # 5.after ReLu(): (BATCH_SIZE, 84)
            # 6.after Linear(): (BATCH_SIZE, 10)

            # so when we calculate loss later, the target must has the same shape as output i.e. (BATCH_SIZE, 10)
            # choosing optimizer should be careful, for example if we want to train a classifer we can use
            # CrossEntropyLoss(), we can pass in a loss with shape torch.size([BATCH_SIZE, 10]) and a label tensor with
            # shape torch.tensor([BATCH_SIZE]) and get a loss with shape torch.size([]). It depends on the algorithm of
            # CrossEntropyLoss(). However we can't use MSELoss() otherwise it will give an error:
            # "The size of tensor a (number of class i.e. 10) must match the size of tensor b (BATCH_SIZE i.e. 128) 
            # at non-singleton dimension 1" This is because MSELoss() needs input x and target y have same shape!
            # if we want to use MSELoss() we can use one-hot representation to generate a label tensor of shape (BATCH_SIZE, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.dense(x)
        return x


def train(net, train_loader, criterion, optimizer, device):
    print("using device: {}".format(device))
    print("start training")
    writer = SummaryWriter(comment="123")
    iter_batch = 0

    for epoch in range(EPOCH):
        net.train() # remember to set train mode in every iteration! https://zhuanlan.zhihu.com/p/302409233
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print("outputs: ", outputs)
            # print("labels: ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # .item() can get the value of one element tensors
            iter_batch += 1
            if i % REPORT_EVERY_BATCH == (REPORT_EVERY_BATCH - 1): # otherwise the first loss will be very small it's not true
                writer.add_scalar("loss", running_loss / REPORT_EVERY_BATCH, iter_batch)
                print("[epoch:%2d, batch: %5d]  loss: %.3f" % (epoch + 1, i + 1, running_loss / REPORT_EVERY_BATCH))
                running_loss = 0.0
    writer.close()
    torch.save(net.state_dict(), PATH)
    print("Done!")


def test(net, test_loader, device):
    print("start testing")

    '''
    net.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training 
    and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them 
    during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation 
    is using torch.no_grad() in pair with model.eval() to turn off gradients computation
    '''
    net.eval()
    net.load_state_dict(torch.load(PATH))
    total, correct = 0, 0

    '''
    Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). 
    It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
    This context manager is thread local; it will not affect computation in other threads.
    '''
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            values, predicted = torch.max(outputs, dim=1) # torch.max() can return max values and their index.
            correct += (predicted == labels).sum().item() # bool value tensor can do sum.
            total += labels.size(0)

    print("Accuracy: %.3f %%" % (100 * correct / total))


def main(to_train=True, to_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true",
        help="Enable cuda computation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)

    '''
    Before training we need to do some transformation. The raw cifar10 images' pixel values range from 0 to 1
    actually it's already ok to train. the pytorch tutorial however normalized pixel values to [-1, 1] by
    using torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), it would be more friendly to
    most neural networks than [0, 1]! As for some explaination, see:https://www.zhihu.com/question/307748349/answer/687791697
    This function expectes a tensor to be a tensor image of size (C, H, W). 
    This transform does not support PIL Image. 
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, 
    this transform will normalize each channel of the input torch.*Tensor:
    output[channel] = (input[channel] - mean[channel]) / std[channel]
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    '''
    if can't download automatically by using torchvision, we can download the tar file manually 
    and go to the cifar.py change the download url as something like this:
    url = "file:///home/your_path/cifar-10-python.tar.gz"
    the "file://" is important!
    '''
    train_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10",
        train=False,
        transform=transform,
        download=True,    
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=LEARNING_RATE, momentum=0.5)
    

    if to_train:
        train(net, train_loader, criterion, optimizer, device)
    if to_test:
        test(net, test_loader, device)


if __name__ == "__main__":
    main(to_train=False, to_test=True)