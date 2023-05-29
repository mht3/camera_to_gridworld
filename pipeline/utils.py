import torch
from skimage.util import random_noise
from torch.utils.data import Dataset
from torchvision import io
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# Must be in the camera_to_gridworld directory


# Global Variables
FILEPATH = "jetbot/data/"
EXT = ".jpg"
GRID_AREA = 9
NUM_IMAGES = len(glob.glob(FILEPATH + "*" + EXT))

class JetbotDataset(Dataset):

    def __init__(self, filepath=FILEPATH, num_images=NUM_IMAGES, grid_area=GRID_AREA, noise=False, transform=None, ext=EXT):
        self.filepath = filepath
        self.ext = ext
        self.grid_area = grid_area
        self.transform = transform
        self.num_images = num_images
        self.noise = noise
        self.images, self.labels = self.load_data()

    def vectorize_label(self, s_label):
        # Converts string label to an array
        label = torch.empty(self.grid_area)
        for i, l in enumerate(s_label):
            label[i] = int(l)
        return label

    def load_data(self):
        '''
        Loads all images and creates corresponding labels.
        '''
        # 178 images in dataset

        # Add room for 178 more noisy images if applicable
        if self.noise: 
            labels = torch.empty(2*self.num_images, self.grid_area)
            images = torch.empty(2*self.num_images, 3, 256, 256)
        else: 
            labels = torch.empty(self.num_images, self.grid_area)
            images = torch.empty(self.num_images, 3, 256, 256)
        i = 0
        for name in glob.glob(self.filepath + "*" + self.ext):
            filename = name[len(self.filepath):]
            # Gets the string label for the 3 x 3 grid, ex: 011001001
            s_label = filename[:self.grid_area]
            # Convert to torch array
            label = self.vectorize_label(s_label)
            labels[i] = label

            image = io.read_image(name).type(torch.float)
            if self.transform:
                image = self.transform(image)
                
            images[i] = image
            if self.noise:
                # Add in images and labels with noise 
                labels[i + self.num_images - 1] = label
                images[i + self.num_images - 1] = torch.tensor(random_noise(image, mode='gaussian', mean=0, var=0.01, clip=True))
            i += 1

        return images, labels

    # Dunder method for length of dataset
    def __len__(self):
        if self.noise:
            return 2*self.num_images
        else:
            return self.num_images
        
    # Dunder method for easy image access during training
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])


def plot_mapper_loss(train_loss, validation_loss):
    epochs = len(train_loss)
    plt.plot(range(1,epochs+1), train_loss, label="Train")
    plt.plot(range(1,epochs+1), validation_loss, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss vs Epoch",size=16)
    plt.savefig("pipeline/results/loss.png", bbox_inches='tight')
    plt.clf()

def plot_mapper_accuracy(train_accuracy, validation_accuracy):
    epochs = len(train_accuracy)
    plt.plot(range(1,epochs+1), train_accuracy, label="Train")
    plt.plot(range(1,epochs+1), validation_accuracy, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch",size=16)
    plt.savefig("pipeline/results/accuracy.png", bbox_inches='tight')
    plt.clf()

def m4e_loss(output, target):
    loss = torch.mean((output - target)**4)
    return loss

def m8e_loss(output, target):
    loss = torch.mean((output - target)**8)
    return loss

def labels_to_images(labels):
    # Reshape to a 3x3 grayscale image. 1 Represents black and 0 represents white
    images = labels.reshape(len(labels), 3, 3)
    return images

'''
Takes in a single 3x3 label and turns it into a grid.
Greyscale image: 0 for white and 1 for black.
See labels_to_images for help.
Example format: 
                arr = np.array([[0., 1., 0.],
                                [1., 0., 0.],
                                [0., 0., 1.]])
'''
def plot_label(label, filename=None):
    plt.imshow(label, cmap="Greys", vmin=0., vmax=1., extent=(0, 3, 0, 3))
    plt.colorbar()

    # Trick to remove ticks but keep grid lines
    plt.xticks(ticks=np.arange(0,4,1),size=0)
    plt.yticks(ticks=np.arange(0,4,1),size=0)
    # Black grid lines to clearly see grid world
    plt.grid(color="black", linestyle='-', linewidth=2)
    if filename != None:
        filepath = "pipeline/results/" + filename + ".png"
        plt.savefig(filepath)
    else:
        plt.show()

def visualize_model(image_ct, orig_images, true_outputs, predicted_outputs, filename=None):
    fig, axs = plt.subplots(image_ct, 3)
    if image_ct != 1:
        for i in range(image_ct):
            axs[i, 0].imshow(orig_images[i])
            axs[i, 0].set_axis_off()  
            axs[i, 1].imshow(true_outputs[i,:,:], cmap="Greys", vmin=0., vmax=1., extent=(0, 3, 0, 3))
            # Trick to remove ticks but keep grid lines
            axs[i, 1].set_xticks(ticks=np.arange(0,4,1))
            axs[i, 1].set_yticks(ticks=np.arange(0,4,1))
            axs[i, 1].tick_params(axis='both', labelsize=0)

            axs[i, 1].grid(color="black", linestyle='-', linewidth=2)

            im = axs[i, 2].imshow(predicted_outputs[i,:,:], cmap="Greys", vmin=0., vmax=1., extent=(0, 3, 0, 3))
            divider = make_axes_locatable(axs[i, 2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
            cbar.ax.tick_params(labelsize=8)

            axs[i, 2].set_xticks(ticks=np.arange(0,4,1))
            axs[i, 2].set_yticks(ticks=np.arange(0,4,1))
            axs[i, 2].tick_params(axis='both', labelsize=0)
            axs[i, 2].grid(color="black", linestyle='-', linewidth=2)

        # Setting Plot Titles
        axs[0, 0].set_title("Input Image")
        axs[0, 1].set_title("True Output")
        axs[0, 2].set_title("Predicted Output")
    else:
        axs[0].imshow(orig_images[0])
        axs[0].set_axis_off()  
        axs[1].imshow(true_outputs[0,:,:], cmap="Greys", vmin=0., vmax=1., extent=(0, 3, 0, 3))
        # Trick to remove ticks but keep grid lines
        axs[1].set_xticks(ticks=np.arange(0,4,1))
        axs[1].set_yticks(ticks=np.arange(0,4,1))
        axs[1].tick_params(axis='both', labelsize=0)

        axs[1].grid(color="black", linestyle='-', linewidth=2)

        im = axs[2].imshow(predicted_outputs[0,:,:], cmap="Greys", vmin=0., vmax=1., extent=(0, 3, 0, 3))
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
        cbar.ax.tick_params(labelsize=8)

        axs[2].set_xticks(ticks=np.arange(0,4,1))
        axs[2].set_yticks(ticks=np.arange(0,4,1))
        axs[2].tick_params(axis='both', labelsize=0)
        axs[2].grid(color="black", linestyle='-', linewidth=2)

        # Setting Plot Titles
        axs[0].set_title("Input Image")
        axs[1].set_title("True Output")
        axs[2].set_title("Predicted Output")

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    if filename != None:
        filepath = "pipeline/results/" + filename + ".png"
        plt.savefig(filepath)
    else:
        plt.show()
    plt.clf()

