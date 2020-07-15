# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss, WeightedPatchLoss, FilteredPatchLoss
from dataset import *
import glob
import torch.optim as optim
import uproot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions())
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=20, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfile", type=str, default="test.root", help='path of .root file for training')
parser.add_argument("--valfile", type=str, default="test.root", help='path of .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model, if applicable")
args = parser.parse_args()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight)
'''
def main():
    machine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=machine)
    model.apply(init_weights)
    criterion = PatchLoss()
    criterion.to(device=machine)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    loss_per_epoch = np.zeros(args.epochs)
    # train the net
    step = 0
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        training_files = glob.glob(os.path.join(args.training_path, '*.root'))
        for training_file in training_files:
            print("Opened file " + training_file)
            branch = get_all_histograms(training_file)
            length = np.size(branch)
            for i in range(length):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                data = get_bin_weights(branch, 0).copy()
                noisy = add_noise(data, args.sigma).copy()
                data = torch.from_numpy(data).to(device=machine)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device=machine)
                out_train = model(noisy.float()).to(device=machine)
                loss = criterion(out_train.squeeze(0).squeeze(0), data, 10)
                loss.backward()
                optimizer.step()
                model.eval()
            model.eval()
        
        # validation
        validation_files = glob.glob(os.path.join(args.validation_path, '*root'))
        epoch_loss = 0
        count = 0
        for validation_file in validation_files:
            print("Opened file " + validation_file)
            branch = get_all_histograms(validation_file)
            length = np.size(branch)
            for i in range (length):
                data = get_bin_weights(branch, 0).copy()
                noisy = add_noise(data, args.sigma).copy()
                data = torch.from_numpy(data).to(device=machine)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device=machine)
                out_train = model(noisy.float()).to(device=machine)
                loss = criterion(out_train.squeeze(0).squeeze(0), data, 10)
                epoch_loss+=loss.item()
            epoch_loss/=length
            count+=1
        # save the model
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
        loss_per_epoch[epoch] = epoch_loss
        print("Average loss per image in epoch " + str(epoch) + " of " + str(args.epochs-1) +": "+ str(epoch_loss))
    loss_plot = plt.plot(loss_per_epoch)
    plt.savefig("loss_plot_july_14.png")
'''
def main():
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using GPU")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = RootDataset(root_file=args.trainfile, sigma = args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)
    val_train = DataLoader(dataset=dataset_val)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = PatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device))
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), 50).to(args.device)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            model.eval()
        training_losses[epoch] = train_loss/len(dataset_train)
        print("t: "+ str(train_loss/len(dataset_train)))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model(val_noise.unsqueeze(1).float().to(args.device))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device), 50).to(args.device)
            val_loss+=output_loss.item()
            #still in progress
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss/len(val_train)
        print("v: "+ str(val_loss/len(val_train)))
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig("Patchloss_plot_July_15.png")

    #make some images and store to csv
    '''
    branch = get_all_histograms("test.root")
    for image in range(10):
        model.to('cpu')
        data = get_bin_weights(branch, image).copy()
        np.savetxt('logs/truth' + str(image) + '.txt', data)
        noisy = add_noise(data, args.sigma).copy()
        np.savetxt('logs/noisy' + str(image) + '.txt', noisy)
        data = torch.from_numpy(data)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.unsqueeze(0)
        noisy = noisy.unsqueeze(1)
        out_train = model(noisy.float()).squeeze(0).squeeze(0)
        np.savetxt('logs/output' + str(image) + '.txt', out_train.detach().numpy())
    '''

def main_2():
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using GPU")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = RootDataset(root_file=args.trainfile, sigma = args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)
    val_train = DataLoader(dataset=dataset_val)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = WeightedPatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device))
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), 50).to(args.device)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            model.eval()
        training_losses[epoch] = train_loss/len(dataset_train)
        print("t: "+ str(train_loss/len(dataset_train)))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model(val_noise.unsqueeze(1).float().to(args.device))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device), 50).to(args.device)
            val_loss+=output_loss.item()
            #still in progress
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss/len(val_train)
        print("v: "+ str(val_loss/len(val_train)))
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig("WeightedPatchloss_plot_July_15.png")
    
def main_3():
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using GPU")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = RootDataset(root_file=args.trainfile, sigma = args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)
    val_train = DataLoader(dataset=dataset_val)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = FilteredPatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device))
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), 50, 0.5).to(args.device)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            model.eval()
        training_losses[epoch] = train_loss/len(dataset_train)
        print("t: "+ str(train_loss/len(dataset_train)))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model(val_noise.unsqueeze(1).float().to(args.device))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device), 50, 0.5).to(args.device)
            val_loss+=output_loss.item()
            #still in progress
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss/len(val_train)
        print("v: "+ str(val_loss/len(val_train)))
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig("Patchloss_plot_July_15.png")



if __name__ == "__main__":
    main()
    main_2()
    main_3()


