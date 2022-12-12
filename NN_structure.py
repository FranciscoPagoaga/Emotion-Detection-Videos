from torch import nn
import torch.nn.functional as F


class model_teamClassifer(nn.Module):
    def __init__(self, ratio_width=4, ratio_height=4, out=32):
        super(model_teamClassifer, self).__init__()
        self.ratio_height = ratio_height
        self.ratio_width = ratio_width
        self.out = out
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # linear layers
        self.linear1 = nn.Linear(512*self.ratio_width*self.ratio_height*4, 768)
        self.linear2 = nn.Linear(768, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, self.out)

        # droput
        self.dropout = nn.Dropout(p=0.1)

        # max_pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Bacth Normalization (conv)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(128)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm6 = nn.BatchNorm2d(512)
        # Batch normalization (linear)
        self.norm_linear1 = nn.BatchNorm1d(768)
        self.norm_linear2 = nn.BatchNorm1d(256)
        self.norm_linear3 = nn.BatchNorm1d(128)

    def drop_last_layer(self, new_out):
        self.out = new_out
        self.linear4 = nn.Linear(128, self.out)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.pool(F.relu(self.norm4(self.conv4(x))))
        x = self.pool(F.relu(self.norm5(self.conv5(x))))
        x = self.pool(F.relu(self.norm6(self.conv6(x))))
        # flattening the image

        x = x.view(8, 512*self.ratio_height*self.ratio_width*4)
        # linear layers
        x = self.dropout(F.relu(self.norm_linear1(self.linear1(x))))
        x = self.dropout(F.relu(self.norm_linear2(self.linear2(x))))
        x = self.dropout(F.relu(self.norm_linear3(self.linear3(x))))
        x = self.linear4(x)

        return x
