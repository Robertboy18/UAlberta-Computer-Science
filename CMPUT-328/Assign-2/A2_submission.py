# Author : Robert Joseph

# Import the required Modules 

import torch
from torchvision import transforms, datasets
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from pprint import pformat

torch.multiprocessing.set_sharing_strategy('file_system')


"""
NoteBooks Referred : CIFAR10_Multiple_Linear_Regression.ipynb
Note : 1 - Documentation has been done below for each function in depth
       2 - Most of the lines have been commented and explained
       3 - Pytorch Documentation was referred
       4 - Reference : https://pytorch.org/docs/stable/
"""


class LogisticRegression(nn.Module):
    """
    Logistic Regression class methods defined here

    Methods : Forward , Constructor
    """
    def __init__(self, params):
        """
        Decorator used  
        Assign the dimension of the dataset and number of classes as the parameters
        Construct the Linear Model by passing it with the parameters in the constructor

        Parameters : params

        Returns : None
        """
        super(LogisticRegression, self).__init__()
        dim = params['dim']
        n_classes = params['n_classes']
        self.Linear = nn.Linear(dim,n_classes)

    def forward(self, x):
        """
        Forward function in the forward pass 
        x.view yields a tensor which gets reshaped 
        Pass the tensor into the Linear Model 
        Calculate the softmax function as this is Multiclass logistic regression

        Parameters : x

        Returns : output (scalar value)

        """
        out = None
        forward_pass = x.view(x.size(0), -1)
        forward_pass = self.Linear(forward_pass)
        out = F.softmax(forward_pass)
        return out

def get_dataset(dataset_name):
    """
    This function gets the dataset that is required either MNIST or CIFAR10
    The parameters are set to each dataset as both are unique
    The MNIST dataset has greyscale images (28*28*1) 
    The CIFAR10 datasets has color images (32*32*3)
    The number of different classes remain the same ie 10
    Reference : MNIST_Multiple_Linear_Regression_Direct.ipynb

    Parameters : dataset_name

    Return: dataset, parameters
    """

    # Intitlaize the dataloaders to None
    train_dataloader = valid_dataloader = test_dataloader = None

    # Batch size for training and testing
    # Powers of 2 are mostly used as its much better after I did batch training dynamics
    # 1024 & 1024 = 92.03 % and 41.45%
    # 128 & 1024 = 92.78
    batch_size_train = 256
    batch_size_test = 1000

    # set parameters mnist
    mnist_params = {
      'dim' : 28*28,
      'n_classes' :10,
      'model':'mnist',
      'learning_rate': 1e-3,
      'optimizer1ma':'adam',
    }

    # set parameters cifar10
    cifar10_params = {
      'dim' : 32*32*3,
      'n_classes' :10,
      'model':'cifar10',
      'learning_rate' : 1e-3,
      'optimizer1ca':'adam', 
      'momentum':0,
      'lambda_val':0
    }

    if dataset_name == "MNIST":
        """
        Get the MNIST Dataset 
        torch.utils.data.Subset was used to get the particular subset ie : 
          Accepts a generator hence why the range(0,number, 1) was used 
            1 - 48000 images for the training set
            2 - The last 12000 images for the validation set
        After that the torch.utils.data.DataLoader was used to load the datasets 
          The batch size that was defined above was used and shuffle = True as each time the dataset is reshuffled at every epoch
        
        """

        # Set the paramters
        params = mnist_params

        # Get the datasets for training and testing 
        MNIST_training = datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
        
        MNIST_test = datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
        
        # Get the various subsets in the dataset
        MNIST_training_dataset = torch.utils.data.Subset(MNIST_training,range(0, 48000, 1))
        MNIST_validation_dataset = torch.utils.data.Subset(MNIST_training,range(48000, 60000, 1))

        # Define the dataloaders that are going to be used for the model
        # num_workers set to 0 as the main process should load the dataloaders compared to many subprocess which could increase time
        train_dataloader  = torch.utils.data.DataLoader(MNIST_training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
        valid_dataloader  = torch.utils.data.DataLoader(MNIST_validation_dataset, batch_size=batch_size_train, shuffle=True,  num_workers=0)
        test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=batch_size_test, shuffle=True, num_workers=0)


    elif dataset_name == "CIFAR10":
        """
        Get the CIFAR10 Dataset 
        torch.utils.data.Subset was used to get the particular subset ie : 
          Accepts a generator hence why the range(0,number, 1) was used 
            1 - 38000 images for the training set
            2 - The last 12000 images for the validation set
        After that the torch.utils.data.DataLoader was used to load the datasets 
          The batch size that was defined above was used and shuffle = True as each time the dataset is reshuffled at every epoch
        
        """

        # Set the paramters
        params = cifar10_params

        # Get the datasets for training and testing
        CIFAR10_training = datasets.CIFAR10("/CIFAR10_dataset/",train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        
        CIFAR10_test = datasets.CIFAR10("/CIFAR10_dataset/",train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        # Get the various subsets in the dataset
        CIFAR10_training_dataset = torch.utils.data.Subset(CIFAR10_training,range(0, 38000, 1))
        CIFAR10_validation_dataset = torch.utils.data.Subset(CIFAR10_training,range(38000, 50000, 1))

        # Define the dataloaders that are going to be used for the model
        # num_workers set to 0 as the main process should load the dataloaders compared to many subprocess which could increase time but is more efficient
        train_dataloader  = torch.utils.data.DataLoader(CIFAR10_training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
        valid_dataloader  = torch.utils.data.DataLoader(CIFAR10_validation_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
        test_dataloader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=batch_size_test, shuffle=True, num_workers=0)
        
    else:
        # raise an error as the dataset specified isn't requested
        raise AssertionError(f'Invalid dataset: {dataset_name}')

    # Assign the dataloaders to the required names ie Train contains the train_dataloder
    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader,
    }

    return dataloaders, params


def test(model, test_dataloader, device, params):
    """
    This function tests the model on the test dataset
    Appends the test predictions to a list 
    Appends the true labels of the images to a list

    Parameters : model , test_dataloader, device , params

    Return : test_predictions , true_labels

    """
    
    # initalize the two lists
    test_predictions = []
    true_labels = []

    # model.eval() switches the model to work in eval mode instead of training mode.
    model.eval()

    # torch.no_grad() speeds up computation and deactivates it with the autograd engine
    # common practise of using both together to speed up 
    # Reference : https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    with torch.no_grad():
        
        # enumerate the test_dataset
        for images, labels in test_dataloader:
            
            # convert from cpu to gpu
            images = images.to(device)
            labels = labels.to(device)

            # view the tensor 
            images = images.view(images.size(0), -1)

            # pass the images to the model and get the output predictions
            output = model(images)

            # get the max element from the predictions 
            # convert the predictions and labels 
            # convert the tensor to numpy 
            index, predictions = torch.max(output.data,dim = 1)  
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            # list appending is faster than numpy appending 
            test_predictions.append(predictions)
            true_labels.append(labels)

    # convert the list to tensors 
    test_predictions = torch.tensor(test_predictions,device = device)
    true_labels = torch.tensor(true_labels,device = device)
    
    return test_predictions, true_labels
    


def validate(model, valid_dataloader, device, params):
    """
    This function perfoms validation on the dataset
    The number of epochs was set to 10
    Find the number of correct prediction on the validation dataset
    Returns the mean accuracy of the validation loss
    
    Parameters : model , valid_dataloader, device , params

    Return : mean_acc

    """
    # initalize the variables
    correct = 0
    mean_acc = 0
      
    # Set all the operations to have no gradient
    with torch.no_grad():
      for images, labels in valid_dataloader:

        # Convert from cpu to gpu
        images = images.to(device)
        labels = labels.to(device)

        # view the tensor
        images = images.view(images.size(0), -1)

        # Find the output prediction 
        output = model(images)

        # get the max element from the predictions 
        index, predictions = torch.max(output.data, 1)

        # Find the validation loss hence 
        # Find the total number of correct predictions
        correct += (predictions == labels).sum().item()      

    # Find the mean validation accuracy loss
    mean_acc = float(correct/len(valid_dataloader))

    # Print the accuracy of the validation set 
    print('\nValidation set: Mean Accuracy: {:.6f}\n'.format(mean_acc))
    
    return mean_acc


def train(model, train_dataloader, valid_dataloader, device, params):
    """
    This function trains the model on the dataset
    First the dataset is selected and the parameters are then set
    Then the optimizer is set 
    Adam was used rather than SGD 
    Adam versus AmsGrad : Reference : https://github.com/schreven/ADAM-vs-AmsGrad
    Reference : http://www.philippeadjiman.com/blog/2018/11/03/visualising-sgd-with-momentum-adam-and-learning-rate-annealing/
    Cross entropy loss is calculated as this a multilabel classification

    Parameters : model , train_dataloader, valid_dataloader, device , params

    Return : mean_train_loss


    """

    # General Parameters
    epochs = 10
    mean_train_loss = 0

    if params['model'] == 'mnist':
      """
      Set the parameters for MNIST
      For the learning rate after hyperparameter tuning the best one was set
      Similar for the Lambda value
      The optimizer was then set with the various parameters and also the L2 regularization term was added
      Referenece : https://www.fast.ai/2018/07/02/adam-weight-decay/
      The optimizer was selected based on the hyperparamter tuning selection model
      """
      learning_rate = params['learning_rate']
      lambda_val_MNIST = 5e-5
      momentum = 0.5

      # If the selected model during hyperparamter tuning is adam then use the Adam optimizer else SGD
      # Default is Adam
      if params['optimizer1ma'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=lambda_val_MNIST,amsgrad=True)
      else:
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=momentum, weight_decay=lambda_val_MNIST,nesterov=True)
    else:
      """
      Set the Paramters for CIFAR
      Default values are chosen if not passed during hyperparemter tuning
      """
      learning_rate = params['learning_rate']

      if params['lambda_val'] == 0:
        lambda_val_CIFAR10 = 5e-6
      else:
        lambda_val_CIFAR10 = params['lambda_val']

      if params['momentum'] == 0:
        params['momentum'] = 0.5
      else:
        momentum = params['momentum']

      # If the selected model during hyperparamter tuning is adam then use the Adam optimizer else SGD
      # Default is Adam
      if params['optimizer1ca'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=lambda_val_CIFAR10,amsgrad=True)
      else:
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=lambda_val_CIFAR10,nesterov=True)
    
    # Loss function defined
    criterion = nn.CrossEntropyLoss()

    # Training the model on a number of epochs
    log_interval = 100
    for epoch in range(epochs):   
 
        model.train()
        for batch_index, (images, labels) in enumerate(train_dataloader): 

            # convert from cpu to gpu
            images = images.to(device) 
            labels = labels.to(device)

            # Clear out the gradients in every call 
            # Else pytorch accumulates it every subsequent call 
            # If we dont use .zero_grad() we wont converge to the required minima
            optimizer.zero_grad()

            # find the output prediction 
            output = model(images)

            # Cross Entropy loss is calculated
            # Regularization is already added in our optimizer 
            # Weight Decay = L2 Regularization
            loss = criterion(output, labels) 
            
            # Back Progragation
            loss.backward()

            # Gradient Descent
            optimizer.step()

            # For each log interval print out the Training details ie : Epoch, Loss 
            # .format formats the output in the specific way we want it to be and 0.6f means 6 values after the decimal
            if (batch_index % log_interval) == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_index * len(images), len(train_dataloader.dataset),
              100. * batch_index / len(train_dataloader), loss.item()))
        
        # Validate our model
        validate(model,valid_dataloader,device,params)


    return mean_train_loss


def tune_hyper_parameter(dataloaders, device, params):
    """
    This function gets the best hyper parameters for the model
    This function uses random search to find the best hyper paramters
    Grid search was used here compared to random search 
    Reference on why on datasets it is almost comparable : https://stats.stackexchange.com/questions/160479/practical-hyperparameter-optimization-random-vs-grid-search
    Accumulate all validation accuracies you compute during hyper parameter search 
    for both optimizers
    Pass the values to be searched and replace it in the Parameters
    Reference for Deep Neural Networks : https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
    Parameters : dataloaders, device, params

    Return : None
    
    """
    best_optimizer = "Adam"
    best_hyperparams = {
        "regularizer": {
            'lambda_val_CIFAR10':0,
        },
        "Adam": {
            "accuracy": 0,
            "learning_rate": 0,
        },
        "SGD": {
            "accuracy": 0,
            "learning_rate": 0,
            "momentum": 0,
        }
    }

    # Adam Optimizer 

    # Randomize each iteration the numbers 
    # Using 64 offset 
    np.random.seed(64)
    adam_accuracy = 0

    # Declare the Grid Value
    grid_value_Adam = {"learning_rate": np.random.uniform(low=0.00001, high=0.0001, size=(3,)),
                      "lambda_val": np.random.uniform(low = 0.003, high = 0.008,size = (2,))}

    # Initalize the validation accuracy list
    validation_accuracy = []

    # Loop through the grid 
    for lr in grid_value_Adam["learning_rate"]:
      for lv in grid_value_Adam['lambda_val']:

        # Define the model 
        model = LogisticRegression(params).to(device)

        # Initalize the Parameters with the values we want to update 
        params['learning_rate'] = lr
        params['optimizer1ca'] = 'adam'
        params['lambda_val'] = lv

        # Train the model on the paramters and find the accuracy on the validation set
        train(model, dataloaders['train'], dataloaders['valid'], device, params)
        accuracy = validate(model, dataloaders['valid'], device, params)
        validation_accuracy.append(accuracy)

        # Find the best values and maximum accuracy and Learning rate as well as the best lambda value
        if accuracy > adam_accuracy:
          best_hyperparams["Adam"]["accuracy"] = accuracy
          best_hyperparams["Adam"]["learning_rate"] = lr
          best_hyperparams["regularizer"]["lambda_val_CIFAR10"] = lv

    # SGD Optimizer

    # Initialize the grid value 
    
    grid_value_sgd = {"learning_rate": np.random.uniform(low=0.00001, high=0.0001, size=(3,)),
                    "momentum": np.random.uniform(low=0.5, high=0.99, size=(2,)),
                    "lambda_val": np.random.uniform(low = 0.003, high = 0.008,size = (1,))}
    
    # initialize the values
    SGD_accuracy = 0
    validation1_accuracy = []

    # Grid Search through
    for lr in grid_value_sgd["learning_rate"]:
      for momentum in grid_value_sgd["momentum"]:
        for lv in grid_value_sgd["lambda_val"]:

          # Define the Model
          model = LogisticRegression(params).to(device)

          # Initialize the Parameters
          params['learning_rate'] = lr
          params['optimizer1ca'] = 'SGD'
          params['momentum'] = momentum
          params['lambda_val'] = lv

          # Train the Model and find the accuracy on the validation set
          train(model, dataloaders['train'], dataloaders['valid'], device, params)
          accuracy = validate(model, dataloaders['valid'], device, params)
          validation1_accuracy.append(accuracy)

          # Find the best paramaters and maximum accuracy
          if accuracy > SGD_accuracy:
            best_hyperparams["SGD"]["accuracy"] = accuracy
            best_hyperparams["SGD"]["learning_rate"] = lr
            best_hyperparams["SGD"]["momentum"] = momentum

            # Only if the accuracy yields better than the Adam accuracy get the best weight decay value
            if accuracy > best_hyperparams["Adam"]["accuracy"]:
              best_hyperparams["regularizer"]["lambda_val_CIFAR10"] = lv
            
    
    # Find the best optimizer
    if best_hyperparams["SGD"]["accuracy"] > best_hyperparams["Adam"]["accuracy"]:
      best_optimizer = "SGD"
      # assign the validation accuracy list of SGD To the validation accuracy list 
      validation_accuracy = validation1_accuracy
    else:
      best_optimizer = "Adam"

    # Print the best results 
    print("\nOptimal performance: Validation Accuracy: {:.3f}, "
            "with {:s} optimizer "
            "using hyper parameters:\n{:s} ".format(
          max(validation_accuracy),
          best_optimizer,
          pformat(best_hyperparams[best_optimizer])))
    
    # Print the best regularization hyper paramters
    print("\nOptimal regularization hyper parameters:\n{:s} ".format(
        pformat(best_hyperparams['regularizer'])))
