import torch
import torchvision
import torchvision.transforms as tfms
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define image preprocessing transformations
image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

mytransforms = tfms.Compose([
    tfms.Resize((image_size, image_size)),
    tfms.ToTensor(),
    tfms.Normalize(imagenet_mean, imagenet_std)
])


# Function to calculate accuracy per feature and confusion matrices for the first X images
def calculate_accuracy_per_feature_and_confusion_matrix(model, dataset, num_images=10):
    feature_names_final = dataset.attr_names[0:40]  # There are only 40 features
    total_features = len(feature_names_final)

    # Initialize counters for accuracy
    correct_predictions = np.zeros(total_features)
    total_predictions = np.zeros(total_features)

    # Initialize confusion matrices
    confusion_matrices = {feature: np.zeros((2, 2)) for feature in feature_names_final}

    for i in range(num_images):
        # Load image and target labels
        img, target_features = dataset[i]
        target_labels = (target_features[0:40] > 0).int().numpy()  # Get first 40 labels and convert to 0/1

        # Preprocess image
        img = torchvision.transforms.ToPILImage()(img)
        img = mytransforms(img)

        # Make predictions
        features = model(img.unsqueeze(0).float())
        predicted_labels = (features > 0).squeeze().int().detach().numpy()

        # Update accuracy counters and confusion matrices
        for idx, label in enumerate(feature_names_final):
            correct_predictions[idx] += (predicted_labels[idx] == target_labels[idx])
            total_predictions[idx] += 1
            confusion_matrices[label] += confusion_matrix([target_labels[idx]], [predicted_labels[idx]], labels=[1, 0])

    # Calculate accuracy per feature
    accuracy_per_feature = {label: correct_predictions[idx] / total_predictions[idx] for idx, label in
                            enumerate(feature_names_final)}

    return accuracy_per_feature, confusion_matrices


# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, feature_name):
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix.astype(int), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 1', 'Predicted 0'],
                yticklabels=['Actual 1', 'Actual 0'])
    plt.title(f'Confusion Matrix for {feature_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# calculate precision and sensitivity
def calculate_precision_sensitivity(confusion_matrices):
    precision_sensitivity = {}
    for feature, conf_matrix in confusion_matrices.items():
        TP = conf_matrix[0, 0]
        FP = conf_matrix[1, 0]
        FN = conf_matrix[0, 1]
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
        precision_sensitivity[feature] = {'precision': precision, 'sensitivity': sensitivity}
    return precision_sensitivity

# plot accuracy per feature
def plot_accuracy_histogram(accuracy_per_feature):
    features = list(accuracy_per_feature.keys())
    accuracies = list(accuracy_per_feature.values())

    plt.figure(figsize=(20, 15))
    plt.bar(features, accuracies, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Feature for the First 10 Images')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.show()


# Usage
data_dir = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA"
weights_celeba = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA\weights\epoch_19_loss_0.256.pt"
model = torch.load(weights_celeba, map_location=torch.device('cpu'))
model.eval()

# Load dataset
dataset = torchvision.datasets.CelebA(data_dir, split='train', target_type=["attr"], transform=mytransforms,
                                      download=False)

# Calculate accuracy per feature and confusion matrices
accuracy_per_feature, confusion_matrices = calculate_accuracy_per_feature_and_confusion_matrix(model, dataset,
                                                                                               num_images=10)
precision_sensitivity = calculate_precision_sensitivity(confusion_matrices)

# print total accuracy
print(f"Total accuracy: {np.mean(list(accuracy_per_feature.values()))}")

# Print accuracy per feature
print("Accuracy per feature:")
for feature, accuracy in accuracy_per_feature.items():
    print(f"{feature}: {accuracy:.2f}")

# Plot accuracy per feature histogram
plot_accuracy_histogram(accuracy_per_feature)

# Plot confusion matrix for a specific feature
specific_feature = "Smiling"
plot_confusion_matrix(confusion_matrices[specific_feature], specific_feature)
print(f"{specific_feature} - Precision: {precision_sensitivity[specific_feature]['precision']:.2f}, \
Sensitivity: {precision_sensitivity[specific_feature]['sensitivity']:.2f}")