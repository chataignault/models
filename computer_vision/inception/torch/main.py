import torch
from inception import SimpleInceptionNet
from dataset import get_cifar10_dataloaders, get_classes
from train import train_model, evaluate_model, save_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = SimpleInceptionNet(num_classes=10)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_dataloaders(batch_size=64, num_workers=2)
    
    print('Starting training...')
    train_accuracies, test_accuracies = train_model(
        model, trainloader, testloader, 
        num_epochs=20, learning_rate=0.001, device=device
    )
    
    final_test_acc = evaluate_model(model, testloader, device)
    print(f'\nFinal test accuracy: {final_test_acc:.2f}%')
    
    save_model(model, 'inception_cifar10.pth')
    
    classes = get_classes()
    print(f'\nCIFAR-10 classes: {classes}')


if __name__ == "__main__":
    main()
