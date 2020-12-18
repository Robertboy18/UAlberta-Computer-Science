# Import librariers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import os
import numpy as np

class Params:
    """
    Reference : Assignment 5
    This class deals with the parameters regarding different evaluaions
    :ivar n_workers: no. of threads for loading data

    :ivar validate_gap: gap in epochs between validations

    :ivar load_weights:
        0: train from scratch,
        1: load and test

    """
    def __init__(self):
        self.n_workers = 0
        self.batch_size = 128
        self.epochs = 1
        self.train_split = 0.76
        self.load_weights = 0
        self.split = 1
        self.lr = 1e-3
        self.wd = 5e-4
        self.alpha = 1
        self.root_path = './yes'
        self.weights_path = os.path.join(self.root_path, './checkpoint')
        self.validate_gap = 100

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
        self.layer.add_module("Bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.layer(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1))
        self.layer1.add_module("Bn", nn.BatchNorm2d(64))
        self.layer1.add_module("Relu", nn.ReLU(True))

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.layer2 = nn.Sequential(
            block(64, 64, 1),
            block(64, 64, 1),
        )

        self.layer4 = nn.Sequential(
            block(64, 128, 2),
            block(128, 128, 1),
        )

        self.layer5 = nn.Sequential(
            block(128, 256, 2),
            block(256, 256, 1),
        )

        self.layer6 = nn.Sequential(
            block(256, 512, 2),
            block(512, 512, 1),
        )

        self.layer7 = nn.Sequential(
            block(512, 1024, 2),
            block(1024, 1024, 1),
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 37*4+20),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        cls = x[:,0:20]
        bb1 = x[:,20:]
        return cls.view(-1, 10, 2), bb1.view(-1, 4, 37)

def validation(model,device,param):
    """
    This function validates the model on the dataset
    Learning Rate Annealing was not used

    Paramaters: device,param,model

    Returns: model
    """

    # Get the datasets 
    valid_x = torch.Tensor(np.load(os.path.join(param.root_path,'valid_X.npy'))).view(5000,1,64,64)
    valid_y = torch.Tensor(np.load(os.path.join(param.root_path,'valid_Y.npy'))).type(torch.LongTensor)
    valid_bboxes = torch.Tensor(np.load(os.path.join(param.root_path,'valid_bboxes.npy'))).type(torch.LongTensor)

    # Use the dataset and then convert it into a dataloader    
    valid_dataset = utils.TensorDataset(valid_x, valid_y, valid_bboxes)
    valid_final = torch.utils.data.Subset(valid_dataset,range(1000, 1100, 1))
    valid_loader = utils.DataLoader(valid_final, batch_size=param.batch_size)

    # Evaluation Mode
    model.eval()

    # Initalize all the variables
    correct = 0
    total = 0
    c = 0

    # Set all the operations to have no gradient
    with torch.no_grad():

        # Iterate over the dataloader
        for images, labels, _ in valid_loader:

            # convert from cpu to gpu
            images = images.to(device)
            labels = labels.to(device)

            # Get the prediced classes and bounding box locations
            classes, bb1 = model(images)

            # get the max class 
            # Reference = https://pytorch.org/docs/stable/generated/torch.sort.html
            classes = torch.sort(classes)[0]
            index, pred = torch.max(classs.data, 1)

            # Reference :https://pytorch.org/docs/stable/generated/torch.sum.html
            corrected = (pred == labels).sum(dim = 1)

            # Get the count of the number of corrected predictons
            for indice in corrected:
                if indice == 2:
                    correct_pred += 1

            # find the accuracy 
            correct += correct_pred
            total += labels.size(0)

    # Print the accuracy 
    accuracy =  100. * correct / total
    print("Accuracy: {:.4f}".format(accuracy))

def train(device, param):
    """
    This function trains and validates the model on the dataset
    Learning Rate Annealing was not used

    Paramaters: device,param

    Returns: model
    """
    
    # Get the datasets 
    train_x = torch.Tensor(np.load(os.path.join(param.root_path,'train_X.npy'))).view(55000,1,64,64)
    train_y = torch.Tensor(np.load(os.path.join(param.root_path,'train_Y.npy'))).type(torch.LongTensor)
    train_bboxes = torch.Tensor(np.load(os.path.join(param.root_path,'train_bboxes.npy'))).type(torch.LongTensor)

    # Use the dataset and then convert it into a dataloader
    train_dataset = utils.TensorDataset(train_x, train_y, train_bboxes)
    train_final = torch.utils.data.Subset(train_dataset,range(0, 100, 1))
    train_loader = utils.DataLoader(train_final, batch_size=param.batch_size)

    # Initalize the model and criterion
    model = ResNet(BasicBlock).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Initalize the optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr = param.lr, weight_decay = param.wd)

    # Train the model over a specific number of epochs
    for epoch in range(param.epochs):
        
        # Set  the gradients
        model.train()

        #Iterate over the dataloader
        for batch_index, (images, labels, bounding_box) in enumerate(train_loader): 

            # convert from cpu to gpu
            images = images.to(device)
            labels = labels.to(device)
            bounding_box = bounding_box.to(device)

             # Clear out the gradients in every call 
            # Else pytorch accumulates it every subsequent call 
            # If we dont use .zero_grad() we wont converge to the required minima
            optimizer.zero_grad()

            # Get the prediced classes and bounding box locations
            classes, bb1 = model(images)

            # Cross Entropy loss is calculated
            # Regularization is already added in our optimizer 
            # Weight Decay = L2 Regularization
            loss_main = criterion(torch.sort(classes)[0], labels)
            losses = []
            for i in range(0,4):
                if i < 2:
                    losses[i] = criterion(bb1[:,i,:], bounding_box[:,0,i])
                else:
                    losses[i] = criterion(bb1[:,i,:], bounding_box[:,1,i-2])
            
            # Get the weighted sum of multiple losses
            loss = loss_main + param.alpha * sum(losses)

            #Back Progragation
            loss.backward()

            # Gradient Descent
            optimizer.step()

            if (batch_index % param.validate_gap) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(images), len(train_loader.dataset),
                100. * batch_index / len(train_loader), loss.item()))

        # Validation set
        validation(model,device,param)

    return model

def test(model,images,pred_class,pred_bboxes):
    """
    This function tests the Res Net Model on the testdataset
    Passes the testdataset to the model 

    Parameters: model,images, pred_class, pred_bboxes, N

    Returns: accuracy,correct1,total1

    """
    # model.eval() switches the net to work in eval mode instead of training mode.
    model.eval()
    #images =  torch.utils.data.Subset(images,range(0, 10, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.no_grad() speeds up computation and deactivates it with the autograd engine
    # common practise of using both together to speed up 
    # Reference : https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    with torch.no_grad():

        # enumerate the test_dataset
        for index, image in enumerate(images):

            #convert the image to a tensor
            image = torch.Tensor(image).to(device)

            # reshape the tensor
            image = image.view(1,1,64,64)

            # pass the images to the model
            classes, bboxes = model(image)

            # get the max class 
            # Reference = https://pytorch.org/docs/stable/generated/torch.sort.html
            classes = torch.sort(classes)[0]
            pred_cls = torch.max(classes.data,dim = 1)[1]

            # Finally assign the  max class
            pred_class[index] = pred_cls.cpu().detach().numpy()

            # get the maximum from each column along dim = 1
            pred = [0]*4
            for i in range(0,4):
                pred[i] = torch.max(bb1[:,i,:].data,dim = 1)[1]

            # Initalize an empty array of 2*4
            # Assign the location values in each row and column from the prediction array 
            # 28 is added to the corners from the inital list ie 
            # [row of the top left corner, column of the top left corner, row of the bottom right corner, column of the bottom right corner
            final_bb = np.empty((2, 4))
            for i in range(0,1):
                for j in range(0,4):
                if i == 0 and j < 2:
                    final_bb[i][j] = pred[i+j]
                elif i == 0 and j > 2:
                    final_bb[i][j] = pred[j-i-2] + 28
                elif i == 1 and j < 2:
                    final_bb[i][j] = pred[i + j + 1]
                else:
                    final_bb[i][j] = pred[i+j-1] + 28

            # Assign the final location
            pred_bboxes[i] = final_bb
    return pred_class, pred_bboxes

def classify_and_detect(images):
    """
    This function is whats called to classify and detect 

    Reference for saving and loading models : https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    :param np.ndarray images: N x 4096 array containing N 64x64 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """

    N = images.shape[0]
    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # Initalize the class Params and get the parameters
    param = Params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if param.load_weights:
        """
        If the model was already saved and only to be tested
        """
        print("-------------Testing after saved model is Loaded---------")
        model = ResNet(BasicBlock)
        device = torch.device('cpu')
        model.load_state_dict(torch.load(param.weights_path, map_location=device))
    else:
        """
        Training from scratch and then saving the model on the CPU
        """
        print("--------------Training from Scratch------------")
        model = train(device, param)
        print("--------------Saving the model-----------------") 
        torch.save(model.state_dict(), param.weights_path)
        device = torch.device('cpu')
        model.to(device)

    # Testing 
    print("---------------Testing the model---------------")

    # Get the predictions
    classes, bboxes = test(model,images,pred_class,pred_bboxes)
    return classes, bboxes

