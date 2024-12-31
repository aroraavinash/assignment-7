from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return acc

    
if __name__ == '__main__':



    """## Data Transformations

    We first start with defining our data transformations. We need to think what our data is and how can we augment it to correct represent images which it might not see otherwise.

    """

    # Train Phase transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    """# Dataset and Creating Train/Test Split"""

    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    """# Dataloader Arguments & Test/Train Dataloaders

    """
    """# Dataloader Arguments & Test/Train Dataloaders

"""

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
    from torchsummary import summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    from model import Model2
    model =  Model2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight decay
    #scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,      # Less aggressive reduction
        patience=3,      # Wait longer before reducing
        min_lr=1e-7     # Allow for smaller learning rates
    )
    summary(model, input_size=(1, 28, 28))
    from tqdm import tqdm
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    EPOCHS = 4
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch)

        accuracy = test(model, device, test_loader)
        scheduler.step(accuracy)