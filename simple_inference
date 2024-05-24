import torch
import torchvision
import torchvision.transforms as tfms
import torchvision.models as models
import numpy as np
import os

image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

mytransforms = tfms.Compose([tfms.Resize((image_size, image_size)),
                             tfms.ToTensor(),
                             tfms.Normalize(imagenet_mean, imagenet_std)])

data_dir = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA"
# train_dataset = torchvision.datasets.CelebA(data_dir, split="train", target_type=["attr"],transform = mytransforms)
train_dataset = torchvision.datasets.CelebA(data_dir, split="train")
feature_names = train_dataset.attr_names
weights_celeba = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA\weights\epoch_19_loss_0.256.pt"

model = torch.load(weights_celeba, map_location=torch.device('cpu'))
# model.load_state_dict(torch.load(weights_celeba, map_location=torch.device('cpu')))
model.eval()
model.to("cpu")


# Function to calculate accuracy per feature
def calculate_accuracy_per_feature(model, image_path, target_labels):
    img = torchvision.io.read_image(image_path)
    img = torchvision.transforms.ToPILImage()(img)
    image = mytransforms(img)
    features = model(image.unsqueeze(0).float())

    # Convert features to binary predictions
    predicted_labels = (features > 0).squeeze().int()

    # Calculate accuracy per feature
    accuracy_per_feature = {}
    feature_names_final = feature_names[0:40]  # There are only 40 features
    total_features = len(feature_names_final)
    correct_predictions = 0
    for idx, label in enumerate(feature_names_final):
        accuracy_per_feature[label] = (predicted_labels[idx] == target_labels[idx]).item()
        if accuracy_per_feature[label]:
            correct_predictions += 1

    overall_accuracy = correct_predictions / total_features

    return accuracy_per_feature, overall_accuracy


img_path = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA\celeba\img_align_celeba\030235.jpg"
img = torchvision.io.read_image(img_path)
img = torchvision.transforms.ToPILImage()(img)
# normalize = tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# composed_transforms = tfms.Compose([Rescale((224, 224)), normalize])
# image = composed_transforms(img)
image = mytransforms(img)
features = model(image.unsqueeze(0).float())

# get feature index where value is greater than 0
feature_idx = np.where(features.detach().numpy() > 0)[1]
features_output = []
for idx in feature_idx:
    features_output.append(feature_names[idx])

# convert to string
features_output = ', '.join(features_output)
print(features_output)

# Checking Accurate per feature
target_labels = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # Need to be filled by hand
accuracy_per_feature, overall_accuracy = calculate_accuracy_per_feature(model, img_path, target_labels)

# Print accuracy per feature
print(f"Overall accuracy: {overall_accuracy}")

'''for feature, accuracy in accuracy_per_feature.items():
    print(f"Accuracy for {feature}: {accuracy}")'''

