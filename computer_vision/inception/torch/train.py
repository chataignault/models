import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)
        
        test_acc = evaluate_model(model, testloader, device)
        test_accuracies.append(test_acc)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - Time: {epoch_time:.2f}s')
    
    return train_accuracies, test_accuracies


def evaluate_model(model, testloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def load_model(model, filepath, device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f'Model loaded from {filepath}')
    return model