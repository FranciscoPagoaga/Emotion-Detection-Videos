from torch import nn
import torch.nn.functional as F


class emotion_classifier(nn.Module):
    def __init__(self, ratio_width=4, ratio_height=4, out=32):
        super(emotion_classifier, self).__init__()
        self.ratio_height = ratio_height
        self.ratio_width = ratio_width
        self.out = out
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # linear layers
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 7)
        # self.linear3 = nn.Linear(128, 7)
        # self.linear4 = nn.Linear(128, self.out)

        # droput
        # self.dropout = nn.Dropout(p=0.1)

        # max_pooling
        self.pool = nn.MaxPool2d(2, 2)

        # # Bacth Normalization (conv)
        # self.norm1 = nn.BatchNorm2d(16)
        # self.norm2 = nn.BatchNorm2d(32)
        # self.norm3 = nn.BatchNorm2d(64)
        # self.norm4 = nn.BatchNorm2d(128)
        # self.norm5 = nn.BatchNorm2d(256)
        # self.norm6 = nn.BatchNorm2d(512)
        # # Batch normalization (linear)
        # self.norm_linear1 = nn.BatchNorm1d(768)
        # self.norm_linear2 = nn.BatchNorm1d(256)
        # self.norm_linear3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
