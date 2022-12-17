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
from tqdm import tqdm


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
        transforms.Resize(32),
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

    validation_images_trasform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Creando Datasets desde ImageLoaders habiendo aplicado las transofrmaciones

    trainset = ImageFolder(root=trainset_route,
                           transform=train_images_transform)
    validationset = ImageFolder(
        root=testset_route, transform=validation_images_trasform)

    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(
        validationset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        

    red = emotion_classifier(out=7)
    if use_cuda:
        red = red.to(device=device)
        print("Utilizando CUDA")
    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=red.parameters(), lr=0.01)
    minimun_valid_loss = np.inf

    training_accurracy_values = np.zeros(0)
    validation_accurracy_values = np.zeros(0)
    
    training_loss_values = np.zeros(0)
    validation_loss_values = np.zeros(0)

    print("Comenzando entrenamiento\n")

    for epoch in range(1, epochs+1):
        print(f'Training epoch {epoch} \n')
        train_loss = 0
        valid_loss = 0
        train_running_correct = 0
        contador_train = 0
        red.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, labels = data
            contador_train += 1
            optimizer.zero_grad()
            if use_cuda:
                image = image.cuda()
                labels = labels.cuda()

            output = red(image)
            loss = criterion(output, labels)
            _, preds = torch.max(output.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'\nValidating epoch {epoch}\n')
        epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
        valid_running_correct = 0
        contador_val = 0
        red.eval()
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            contador_val += 1
            image, labels = data
            optimizer.zero_grad()
            if use_cuda:
                image = image.cuda()
                labels = labels.cuda()

            output = red(image)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            valid_running_correct += (preds == labels).sum().item()

        epoch_acc_val = 100. * (valid_running_correct /
                                len(valid_loader.dataset))

        auxLoss_valid = valid_loss/contador_val
        auxLoss_train = train_loss/contador_train
        print(
            f'Epoch {epoch} \tTraining Loss: {round(auxLoss_train,6)}\tValidation Loss: {round(auxLoss_valid, 6)}\n\tTraining Accurracy: {round(epoch_acc, 6)}\tValidation Accurracy:{epoch_acc_val}')
        print("\n")
        
        training_loss_values = np.append(training_loss_values, auxLoss_train)
        training_accurracy_values = np.append(training_accurracy_values, epoch_acc)
        validation_loss_values = np.append(validation_loss_values, auxLoss_valid)
        validation_accurracy_values = np.append(validation_accurracy_values, epoch_acc_val)

        print("="*200)
        if auxLoss_valid <= minimun_valid_loss:
            fails = 0
            print(
                f'Validation loss decreased from {round(minimun_valid_loss, 6)} to {round(auxLoss_valid, 6)}')
            torch.save(red.state_dict(), 'modeloRed_3.pkl')
            minimun_valid_loss = auxLoss_valid
            print('Saving New Model')
            print("="*200)
        else:
            # si las fallas llega a 30, se cierra el programa y se guarda el modelo
            fails += 1
            if fails >= 30:
                print('Loss haven\'t decrease in a time! Saving Last Model')
                torch.save(red.state_dict(), 'modeloRed_3.pkl')
                minimun_valid_loss = auxLoss_valid
                exit(0)

    
    torch.save(red.state_dict(), 'modeloRed_3.pkl')
    print('Saving New Model')
    print("="*200)
if __name__ == "__main__":
    main()
