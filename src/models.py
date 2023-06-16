import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net_1(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )
        self.pool =  nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1),
            nn.ReLU()
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.conv1x1(x))
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.adaptive_avg_pool(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)


class Net_2(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU()
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU()
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1)
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1x1_1(x))
        x = self.conv2(x)
        # print(x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)

        x = F.relu(self.conv1x1_2(x))

        x = self.conv5(x)
        x = self.conv6(x)

        x = F.relu(self.conv1x1_3(x))
        x = self.adaptive_avg_pool(x)
        x = x.squeeze()

        return F.log_softmax(x, dim=1)


class Net_3(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        # pool
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1)
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1x1_1(x))
        x = self.conv2(x)
        # print(x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)

        x = F.relu(self.conv1x1_2(x))

        x = self.conv5(x)
        x = self.conv6(x)

        x = F.relu(self.conv1x1_3(x))
        x = self.adaptive_avg_pool(x)
        x = x.squeeze()

        return F.log_softmax(x, dim=1)


class Net_4(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            # ,nn.Dropout2d(0.1)  

        )
        # pool
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
            # nn.Dropout2d(0.1)          
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1)
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1x1_1(x))
        x = self.conv2(x)
        # print(x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        # x = self.pool(x)

        x = F.relu(self.conv1x1_2(x))

        x = self.conv5(x)
        x = self.conv6(x)

        # x = F.relu(self.conv1x1_3(x))
        x = self.adaptive_avg_pool(x)
        x = self.conv1x1_3(x)
        x = x.squeeze()

        return F.log_softmax(x, dim=1)
    
    
def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)
