import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
# Need the CNN class to load the mapper model
from mapper import CNN, JetbotDataset
import argparse


'''
Loads pretrained mapper model and plots results of the form: (transformed image, original label, predicted label)
'''
def visualize_results(image_ct=3):
    model = torch.load("pipeline/mapper_model.pt")
    data = JetbotDataset(filepath=utils.FILEPATH, num_images=utils.NUM_IMAGES, transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]))
    data_loader = DataLoader(data, batch_size=image_ct, shuffle=True, generator=torch.Generator().manual_seed(42))

    # Create 3 arrays of images, labels, and predicted labels
    orig_images = []
    true_labels = None
    predicted_labels = None
    # Grabs images in first batch of data_loader
    for batch_idx, (images, label)  in enumerate(data_loader):
        if batch_idx == 0:
            # Get output from pretrained model
            output = model(images)
            # Convert to numpy array of (height, width, channels) for plotting
            for image in images:
                image = image.permute(1, 2, 0).numpy()
                orig_images.append(image)

            # Add true label
            true_labels = label.numpy()
            # Label predicted by model
            predicted_labels = output.detach().numpy()
            break

    
    true_outputs = utils.labels_to_images(true_labels)
    predicted_outputs = utils.labels_to_images(predicted_labels)
    utils.visualize_model(image_ct, orig_images, true_outputs, predicted_outputs, filename="Model_Output")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Argument for number of images to use for visualization. 
    # To use with 5 images, run: python pipeline/evaluate_model.py 5
    parser.add_argument("-i", "--images", default=3, type=int, help="Enter number of images for visualization.")
    args = parser.parse_args()
    visualize_results(image_ct=args.images)