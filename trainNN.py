import torch
import sys
import pickle
import numpy as np
from os import listdir
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from NN_structure import emotion_classifier


def main():
    trainset_route = sys.argv[1]
    testset_route = sys.argv[2]
    epochs = int(sys.argv[3])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = False
    if device == 'cuda':
        use_cuda = True
    BATCH_SIZE = 16

    # trasnforming images and converting (and normalizang) them into tensors so we can send them to the emotions classifier
    # Procedimiento encontrado en https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/

    # Aplicando filtros y aumenbntos para variar el entrenamiento y los datos y tratar de evitar el overfitting

    train_images_transform = transforms.Compose([
        transforms.Resize(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Aqui ya no se necesita aplicar muchos filtros pues esos son para mejorar el entrenamiento de la red, solo debemos convertir a tensor

    validation_images_trasform = transforms.Compose({
        transforms.Resize(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    })

    # Creando Datasets desde ImageLoaders habiendo aplicado las transofrmaciones

    trainset = ImageFolder(root=trainset_route,
                           transform=train_images_transform)
    validationset = ImageFolder(
        root=testset_route, transform=validation_images_trasform)

    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    red = emotion_classifier(out=7)
    if use_cuda:
        red = red.to(device=device)
        print("Utilizando CUDA")
    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=red.parameters(), lr=0.02)

    minimun_valid_loss = np.inf

    print("Comenzando entrenamiento")

    for epoch in range(1, epochs+1):
        print(f'Training {epoch}')
        train_loss = 0
        valid_loss = 0
        red.train()
        for (data, target) in train_loader:
            optimizer.zero_grad()
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            output = red(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        print(f'Validating {epoch}')
        red.eval()
        for (data, target) in valid_loader:
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            output = red(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        auxLoss_valid = valid_loss/(len(valid_loader)*BATCH_SIZE)
        auxLoss_train = train_loss/(len(train_loader)*BATCH_SIZE)

        print(
            f'Epoch {epoch} \tTraining Loss: {round(auxLoss_train,6)}\tValidation Loss: {round(auxLoss_valid, 6)}')

        if auxLoss_valid <= minimun_valid_loss:
            fails = 0
            print(
                f'Validation loss decreased from {round(minimun_valid_loss, 6)} to {round(auxLoss_valid, 6)}')
            torch.save(red.state_dict(), 'modeloRed.pkl')
            minimun_valid_loss = valid_loss
            print('Saving New Model')
            print("="*100)
        else:
            # si las fallas llega a 10, se cierra el programa y se guarda el modelo
            fails += 1
            if fails >= 100:
                print('Loss haven\'t decrease in a time! Saving Last Model')
                torch.save(red.state_dict(), 'modeloRed.pkl')
                minimun_valid_loss = auxLoss_valid
                exit(0)


if __name__ == "__main__":
    main()
