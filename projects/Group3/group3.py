import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import random

# Define the stuffs
IMAGE_DIRECTORY = '/home/group3/projects/beeml/dataset/thefinal/wboc/train'
SAVE_DIRECTORY = '/home/group3/projects/beeml/cnn2'
DATASET_NAME = 'wboc'
IMAGE_SIZE = (224, 224)  # Input image size
AUGMENTATION_FACTOR = 2  # Number of augmented samples per data point

# Check if CUDA is available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Found {num_devices} CUDA device(s).")
else:
    print("CUDA is not available. Using CPU.")

def define_transformations():
    # Common transformations applied to all images
    common_transforms = [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
    
    return transforms.Compose(common_transforms)

def define_augmentations():
    # Additional transformations for augmentation
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        # Add more transformations as needed
    ]
    
    return transforms.Compose(augmentation_transforms)

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load images from directory with filenames and integer labels
def load_images(directory):
    data = []
    labels = []
    image_filenames = []  # Store image filenames here
    class_indices = {}  # Map class names to integer labels
    index = 0
    for class_name in tqdm(os.listdir(directory), desc="Loading images"):
        class_dir = os.path.join(directory, class_name)
        class_indices[class_name] = index  # Assign an integer label to each class
        for image_file in tqdm(os.listdir(class_dir), desc=f"Class: {class_name}"):
            image_path = os.path.join(class_dir, image_file)
            if os.path.isfile(image_path) and image_file.endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(image_path).convert('RGB')
                data.append(image)
                labels.append(index)  # Append the integer label
                image_filenames.append(image_file)  # Store the image filename
        index += 1
    return data, labels, image_filenames, class_indices

# Perform train-test split
def perform_train_test_split(data, labels, test_size=0.2, random_state=42):
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)

# Augment data
def augment_data(data, labels, augmentation_factor, transform):
    augmented_data = []
    augmented_labels = []

    # Determine the size of the majority class
    class_counts = {label: labels.count(label) for label in set(labels)}
    majority_class_size = max(class_counts.values())

    # Augment each class
    for class_label, class_size in class_counts.items():
        # Determine the target size after augmentation
        target_size = max(majority_class_size * augmentation_factor, class_size)
        
        # Augment the class to match the target size
        class_indices = [i for i, label in enumerate(labels) if label == class_label]
        num_augmented_samples = target_size - class_size
        for _ in tqdm(range(num_augmented_samples), desc=f"Augmenting class {class_label}"):
            idx = random.choice(class_indices)
            image = data[idx]
            augmented_image = transform(image)
            augmented_data.append(augmented_image)
            augmented_labels.append(class_label)

    # Combine original data with augmented data
    augmented_data += data      # Include original data
    augmented_labels += labels  # Include original labels

    # Print the final size of each class after augmentation
    print("Size of each class after augmentation:")
    for class_label in set(augmented_labels):
        class_size = augmented_labels.count(class_label)
        print(f"Class {class_label}: {class_size}")

    return augmented_data, augmented_labels

# Create custom datasets
def create_datasets(X_train_data, X_test_data, y_train, y_test, transform):
    train_dataset = CustomDataset(X_train_data, y_train, transform=transform)
    test_dataset = CustomDataset(X_test_data, y_test, transform=transform)
    return train_dataset, test_dataset

# Create data loaders
def create_data_loaders(train_dataset, test_dataset, batch_size=32, shuffle=True):
    num_workers = mp.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the input size of the fully connected layer dynamically
        self.fc_input_size = self.calculate_fc_input_size(input_size)
        
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(-1, self.fc_input_size)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def calculate_fc_input_size(self, input_size):
        # Calculate the size of the feature maps after passing through convolutional and pooling layers
        conv_output_size = np.array(input_size)
        for _ in range(5):  # Five convolutional layers
            conv_output_size = np.floor((conv_output_size - 3 + 2 * 1) / 1) + 1  # Using kernel_size=3, padding=1, stride=1
            conv_output_size = np.floor((conv_output_size - 2) / 2) + 1  # Using MaxPool2d with kernel_size=2, stride=2
        fc_input_size = int(conv_output_size[0] * conv_output_size[1] * 512)
        return fc_input_size

# Initialize model, loss function, and optimizer
def initialize_model(num_classes, device_ids):
    model = CNNModel(num_classes=num_classes, input_size=IMAGE_SIZE).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)  # Utilize multiple GPUs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return model, criterion, optimizer

# Train model
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluate model
def evaluate_model(model, test_loader, test_image_filenames, class_indices):
    model.eval()
    correct = 0
    total = 0
    misclassified_images = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            misclassified_indices = (predicted != labels).nonzero().squeeze().cpu().numpy()
            if len(misclassified_indices.shape) == 0:
                misclassified_indices = [misclassified_indices.item()]  # Convert to list if it's a scalar
            for idx in misclassified_indices:
                filename = test_image_filenames[idx]
                actual_label = labels[idx].item()
                predicted_label = predicted[idx].item()
                actual_class_name = list(class_indices.keys())[list(class_indices.values()).index(actual_label)]
                predicted_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_label)]
                misclassified_images.append([filename, actual_class_name, predicted_class_name])
    accuracy = correct / total
    print(f"Test Accuracy,{DATASET_NAME}: {accuracy:.4f}")
    return misclassified_images

# Save model
def save_model(model, directory, dataset_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_filename = f"{dataset_name}_cnn2.pth"
    torch.save(model.state_dict(), os.path.join(directory, model_filename))
    return model_filename

# Main function
def main():
    # Load images from directory
    data, labels, image_filenames, class_indices = load_images(IMAGE_DIRECTORY)

    # Perform train-test split
    X_train_data, X_test_data, y_train, y_test = perform_train_test_split(data, labels)

    # Convert labels to tensor
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Define transformations
    common_transform = define_transformations()
    augmentation_transform = define_augmentations()

    # Augment training data
    augmented_data, augmented_labels = augment_data(X_train_data, y_train.tolist(), augmentation_factor=AUGMENTATION_FACTOR, transform=augmentation_transform)

    # Create custom datasets with common transformations only
    train_dataset, test_dataset = create_datasets(augmented_data, X_test_data, augmented_labels, y_test, common_transform)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)

    # Initialize model, loss function, and optimizer
    model, criterion, optimizer = initialize_model(num_classes=len(class_indices), device_ids=[1, 2, 3])

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Evaluate the model and get misclassified images
    misclassified_images = evaluate_model(model, test_loader, image_filenames, class_indices)

    # Save the trained model
    saved_model_filename = save_model(model, SAVE_DIRECTORY, DATASET_NAME)

    # Write misclassified images to a CSV file
    csv_filename = f"{DATASET_NAME}_misclassified.csv"
    csv_file_path = os.path.join(SAVE_DIRECTORY, csv_filename)
    df = pd.DataFrame(misclassified_images, columns=['Filename', 'Actual Class', 'Predicted Class'])
    df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    main()
