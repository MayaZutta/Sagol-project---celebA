import numpy as np
import torchvision
import torchvision.transforms as tfms
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

transform = tfms.Compose([tfms.Resize((image_size, image_size)),
                          tfms.ToTensor(),
                          tfms.Normalize(imagenet_mean, imagenet_std)])

# Download and load the CelebA dataset - target_type=["identity"] ensures that the identity labels are loaded
data_dir = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA"
dataset = torchvision.datasets.CelebA(data_dir, split='train', target_type=["identity"], transform=transform,
                                      download=False)

# Extract identity labels
identity_labels = dataset.identity

identity_to_images = defaultdict(list)
for idx, identity in enumerate(identity_labels):
    identity = identity.item()
    identity_to_images[identity].append(idx)

# Find the identity with the maximum number of images and some debugging
''' max_images_identity = max(identity_to_images, key=lambda identity: len(identity_to_images[identity]))
max_images_count = len(identity_to_images[max_images_identity])

for identity, indices in identity_to_images.items():
    print(f"Identity {identity}: {len(indices)} images")'''

# Select the two identities with at least 30 images each
selected_identities = [identity for identity, indices in identity_to_images.items() if len(indices) >= 30][:2]

# Ensure we have exactly 100 images per selected identity
sampled_images = {identity: identity_to_images[identity][:30] for identity in selected_identities}

print(f"Selected Identities: {selected_identities}")
print(f"Number of images for each selected identity: {[len(sampled_images[i]) for i in selected_identities]}")

# Function to show images
def show_images(images, title):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for img, ax in zip(images, axes):
        ax.imshow(tfms.ToPILImage()(img))
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize the selected identities
for identity in selected_identities:
    images_to_show = [dataset[i][0] for i in sampled_images[identity][:3]]  # Show first 3 images
    show_images(images_to_show, f"Images for Identity {identity}")

# Prepare data for t-SNE
all_images = []
all_labels = []
for identity in selected_identities:
    for idx in sampled_images[identity]:
        img, _ = dataset[idx]
        all_images.append(img.numpy().flatten())
        all_labels.append(identity)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(all_images)

# Plot t-SNE result
plt.figure(figsize=(10, 8))
for identity in selected_identities:
    indices = np.where(all_labels == identity)
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Identity {identity}')

plt.title('t-SNE Visualization of Selected Identities')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

