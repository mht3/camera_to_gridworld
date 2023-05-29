import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
# Need the CNN class to load the mapper model
from mapper import CNN, JetbotDataset


'''
Loads pretrained mapper model and plots results of the form: (transformed image, original label, predicted label)
'''
def visualize_avg():
    data = JetbotDataset(filepath=utils.FILEPATH, num_images=utils.NUM_IMAGES, transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]))
    data_loader = DataLoader(data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(36))

    # Create 3 arrays of images, labels, and predicted labels
    orig_images = []
    true_labels = None
    predicted_labels = None
    # Grabs images in first batch of data_loader
    label_list = []
    for batch_idx, (images, label)  in enumerate(data_loader):
        label_list.append(label)

    labels = torch.cat(label_list)
    avg = torch.mean(labels, dim=0)
    
    # outputs = utils.labels_to_images(avg)
    # utils.plot_label(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Argument for number of images to use for visualization. 
    # To use with 5 images, run: python pipeline/evaluate_model.py 5
    visualize_results()