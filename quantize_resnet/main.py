import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from quanti import *
from model.resnet18 import *


if __name__ == "__main__":

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="../kaixin.sun/Cifar10_data", train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=20
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="../kaixin.sun/Cifar10_data", train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=20
    )

    criterion = torch.nn.CrossEntropyLoss()
    #指定当前模式
    print("build and train the net without bn......")
    mode = QuantiMode.kTrainingWithoutBN
    net = ResNet18(Basicmode =mode)
    
    optimizer = optim.Adam(net.parameters())

    device = torch.device("cuda")

    for epoch in range(20):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            length = len(trainloader)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total
            if (batch_idx + 1 + epoch*length)%10 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '%(epoch+1,(batch_idx + 1 + epoch*length),sum_loss / (batch_idx+1),100. * acc))

        print("Waiting Test...")
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct / total
        print("epoch: {}, test_acc: {}".format(epoch, acc))
    

    '''
    print("build the net and test......")
    mode = QuantiMode.kIntInference
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            net._init_inference()
            outputs = net.
    '''
    





