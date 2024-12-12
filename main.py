
import numpy as np

#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

#Visulization
import matplotlib.pyplot as plt

#Others
import time
import copy
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('data/', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

class_names = trainset.classes
dataset_sizes = {'train':len(trainset), 'val':len(valset)}
dataloaders  = {'train':trainloader, 'val':valloader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataiter = iter(trainloader)
images, labels = next(dataiter)

print(type(images))
print(images.shape)
print(labels.shape)

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )

        self.fcs = nn.Sequential(
            nn.Linear(2304, 1152),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1152, 576),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(576, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
model = Net()
#print(model)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if torch.cuda.is_available():
    model = model.to(device)
    criterion = criterion.to(device)
def train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs=11):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    print(f'Best validation accuracy: {best_acc:.4f}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# Train the model
model_ft, train_loss, val_loss, train_acc, val_acc = train_model(
    model, criterion, optimizer, exp_lr_scheduler, dataset_sizes, dataloaders, num_epochs=10
)

# Plot loss and accuracy
epochs = range(1, len(train_loss) + 1)
plt.figure()
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.plot(epochs, train_acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Loss/Accuracy graph: baseline')
plt.legend()
plt.show()

class_names = [str(i) for i in range(10)]

# Adjust imshow function
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().squeeze()  # Ensure tensor is on CPU
    if inp.ndim == 3:  # Multi-channel image
        inp = inp.transpose((1, 2, 0))  # Convert to HxWxC
    inp = (inp - inp.min()) / (inp.max() - inp.min())  # Normalize to [0, 1]
    plt.imshow(inp, cmap='gray_r' if inp.ndim == 2 else None)
    if title is not None:
        plt.title(title)

# Adjust visualize_model
def visualize_model(model, num_images=6):
    num_images = (num_images // 2) * 2  # Ensure even number of images
    was_training = model.training  # Save current mode
    model.eval()  # Switch to evaluation mode
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                imshow(inputs[j].cpu())  # Ensure tensor is on CPU
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    model.train(mode=was_training)  # Restore mode
                    return
        plt.tight_layout()
        plt.show()
        model.train(mode=was_training)  # Restore mode

visualize_model(model)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions for the validation set
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot confusion matrix
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion matrix: baseline')
plt.show()



import torch.nn.functional as F

def report_misclassified_results(model, dataloader, num_images=6):
    """
    Report misclassified images with their true labels, predicted labels, and confidence values.

    Args:
    - model: Trained CNN model.
    - dataloader: Validation dataloader.
    - num_images: Number of misclassified images to display.

    Returns:
    - None (displays images with their predictions, true labels, and confidence).
    """
    model.eval()  # Set model to evaluation mode
    incorrect_images = []
    incorrect_confidences = []
    incorrect_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, preds = torch.max(outputs, 1)  # Predicted labels

            for i in range(inputs.size(0)):
                if preds[i] != labels[i]:  # Only collect misclassified samples
                    confidence_value = confidences[i, preds[i]].item()
                    incorrect_images.append(inputs[i].cpu())
                    incorrect_labels.append(labels[i].item())
                    predicted_labels.append(preds[i].item())
                    incorrect_confidences.append(confidence_value)

                # Stop if we have enough images
                if len(incorrect_images) >= num_images:
                    break
            if len(incorrect_images) >= num_images:
                break

    # Plot misclassified samples
    print("Misclassified images:")
    plt.figure(figsize=(12, 6))
    for i in range(len(incorrect_images)):
        ax = plt.subplot(2, (num_images + 1) // 2, i + 1)
        imshow(incorrect_images[i], title=f"True: {class_names[incorrect_labels[i]]}\n"
                                          f"Pred: {class_names[predicted_labels[i]]}\n"
                                          f"Conf: {incorrect_confidences[i]:.2f}")
    plt.tight_layout()
    plt.show()

# 调用函数以生成报告
report_misclassified_results(model, dataloaders['val'], num_images=6)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')



