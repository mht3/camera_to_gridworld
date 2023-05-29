import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm 
from sklearn.metrics import accuracy_score
from utils import JetbotDataset, plot_mapper_loss, plot_mapper_accuracy, labels_to_images, visualize_model
from utils import FILEPATH, NUM_IMAGES
import argparse


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(6, 6), stride=(2, 2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(6, 6), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 6), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(6, 6), stride=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 6), stride=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(6, 6), stride=(1, 1))

        self.fc1 = nn.Linear(in_features= 1024, out_features=256)
        # out features 3x3 grid
        self.fc2 = nn.Linear(in_features=256, out_features=9)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        # Flatten everything but batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Gridworld_Mapper():
    def __init__(self, filepath, num_images, noise=False):
        self.filepath = filepath
        self.num_images = num_images
        self.threshold = 0.7
        self.noise = noise
            
        # Architecture from CNN class defined above
        self.cnn = CNN()

    def train(self, batch_size=64, epochs=100, lr=0.0001):
        print("Preprocessing dataset ...")
        dataset = JetbotDataset(filepath=self.filepath, num_images=self.num_images, noise=self.noise , transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]))
        self.num_images = len(dataset)
        pct_validation = 0.15
        num_validation = int(self.num_images*pct_validation)
        num_train = self.num_images - num_validation
        train_data, validation_data = torch.utils.data.random_split(dataset, [num_train, num_validation], generator=torch.Generator().manual_seed(43))
        print(f"...\nDone.\n{len(train_data)} training data entries.\n{len(validation_data)} validation data entries.\n")

        # Load data with random seed so results are reproducible.
        # data comes in as (batch_size, channels, height, width)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
        validation_loader = DataLoader(dataset=validation_data, batch_size=num_validation, shuffle=True, generator=torch.Generator().manual_seed(41))

        # Adam optimizer
        optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        # Loss function
        criterion = nn.MSELoss(reduction='mean')
        train_loss = []
        validation_loss = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, epochs+1):

            # Set model to training
            self.cnn.train()
            # track running total of losses in each epoch
            total_loss = 0.
            train_acc = 0.
            # train loader data with progress bar
            tl = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (images, targets) in tl:
                # Zero out gradients for this batch
                optimizer.zero_grad()
                # Get predicted labels
                outputs = self.cnn(images)
                # Get loss
                loss = criterion(outputs, targets)
                # Back Prop
                loss.backward()
                # Update weights
                optimizer.step()

                # Get accuracy
                rounded_outputs = (outputs > self.threshold).float()
                acc = accuracy_score(targets, rounded_outputs)
                # Update progress bar
                tl.set_description(f"Epoch [{epoch}/{epochs}]")
                tl.set_postfix(loss=loss.item(), acc=acc)
                # Add to running loss
                total_loss += loss.item()
                # Add to running accuracy
                train_acc += acc
                if epoch % 10 == 0 and batch_idx==0:
                    # Add true label
                    true_labels = targets.numpy()
                    # Label predicted by model
                    predicted_labels = outputs.detach().numpy()
                    true_outputs = labels_to_images(true_labels)
                    predicted_outputs = labels_to_images(predicted_labels)
                    image_group = []
                    for i in range(len(images)):
                        image = images[i].permute(1, 2, 0).numpy()
                        image_group.append(image)
                    visualize_model(3, image_group, true_outputs[0:3], predicted_outputs[0:3], filename="Output_Epoch_{}".format(epoch))
            
            avg_loss =  total_loss/len(train_loader)
            train_loss.append(avg_loss)
            avg_acc = train_acc/len(train_loader)
            train_accuracy.append(avg_acc)
            # Set model to evaluate
            self.cnn.eval()
            val_loss = 0.
            val_acc = 0.
            for batch_idx, (images, targets) in enumerate(validation_loader):
                outputs = self.cnn(images)
                # Get loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                # Get accuracy
                rounded_outputs = (outputs > self.threshold).float()
                val_acc += accuracy_score(targets, rounded_outputs)
                if epoch % 10 == 0 and batch_idx==0:
                    # Add true label
                    true_labels = targets.numpy()
                    # Label predicted by model
                    predicted_labels = outputs.detach().numpy()
                    true_outputs = labels_to_images(true_labels)
                    predicted_outputs = labels_to_images(predicted_labels)
                    image_group = []
                    for i in range(len(images)):
                        image = images[i].permute(1, 2, 0).numpy()
                        image_group.append(image)
                    visualize_model(3, image_group, true_outputs[0:3], predicted_outputs[0:3], filename="Validation_Output_Epoch_{}".format(epoch))

            avg_valid_loss = val_loss/len(validation_loader)
            validation_loss.append(avg_valid_loss)
            avg_val_acc = val_acc/len(validation_loader)
            validation_accuracy.append(avg_val_acc)
            if epoch % 10 == 0:
                print("Epoch: {}\t Loss: {:.4f}\t Val Loss: {:.4f}\t Acc: {:.4f}\t Val Acc: {:.4f}".format(epoch, avg_loss, avg_valid_loss, avg_acc, avg_val_acc))

        print("Training Complete")
        return train_loss, validation_loss, train_accuracy, validation_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments for epochs and batch size. 
    parser.add_argument("-b", "--batch_size", type=int, help="Enter batch size for model.", default=32)
    parser.add_argument("-e", "--epochs", type=int, help="Enter number of epochs.", default=100)

    args = parser.parse_args()

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = Gridworld_Mapper(filepath=FILEPATH, num_images=NUM_IMAGES)
    train_loss, validation_loss, train_accuracy, validation_accuracy = mapper.train(epochs=args.epochs, batch_size=args.batch_size, lr=0.0001)
    plot_mapper_loss(train_loss, validation_loss)
    plot_mapper_accuracy(train_accuracy, validation_accuracy)
    torch.save(mapper.cnn, "pipeline/mapper_model.pt")