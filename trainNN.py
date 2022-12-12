import torch
import sys
import numpy as np
from os import listdir
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder

def main():
    trainset_route = sys.argv[1]
    testset_route = sys.argv[2]
    trainset = ImageFolder(root=trainset_route)
    print(trainset.classes)

if __name__=="__main__":
    main()