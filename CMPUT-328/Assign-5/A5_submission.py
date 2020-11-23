import torch
import pandas as pd
import torch.nn as nn

from A5_utils import REAL_UNLABELED, FAKE_UNLABELED, REAL_LABEL, FAKE_LABEL


class TrainParams:
    """
    :ivar n_workers: no. of threads for loading data

    :ivar validate_gap: gap in epochs between validations

    :ivar tb_path: folder where to save tensorboard data

    :ivar load_weights:
        0: train from scratch,
        1: load and test
        2: load if it exists and continue training

    """

    def __init__(self):
        self.n_workers = 0
        self.batch_size = 128
        self.n_epochs = 10
        self.load_weights = 1
        self.tb_path = './tensorboard'
        self.weights_path = './checkpoints/model.pt.1'
        self.validate_gap = 10


class OptimizerParams:
    def __init__(self):
        self.type = 0
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.beta1 = 0.5


class SharedNet(nn.Module):
    """
    Shared Net Class
    SharedNet contains the shared layers between Discriminator and Classifier and its output is used as input to both
    The number of Parameters used in this class are 6310739.
    This class is what does the semi supervised learning and the Classifier learns the information and gets better 
    Similarly the Discriminator learns to get better at becoming a so called better detective that can distinguish the fake 
    images from the Generator

    Functions : forward, init_weight

    """
    class Params:
        """
        Inherited Class
        Dummy variable set to 0 as it inherits no Parameters

        Functions : None

        """
        dummy = 0

    def __init__(self, params, n_channels=3, img_size=32):
        """
        Constructor Class 
        The constructor initializes the model with the required CNN layers
        Definition of the CNN Net we are using and all the layers
        Batch Normalization was used for most of the layers
        Dropout was also used to prevent Overfitting
        All the CNN layers are weight initalized by the init_weights function
        
        params : params, n_channels, img_size

        returns : None

        """
        super(SharedNet, self).__init__()

        # CNN + Batch Normalization 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16,kernel_size = 3,stride = 1,padding = 1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = 3,stride = 1,padding = 1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size=3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(1024)
        self.conv8 = nn.Conv2d(in_channels = 1024, out_channels = 3, kernel_size=3, stride=1, padding=1)

        # Dropout Layer
        self.Dropout = nn.Dropout(0.2)

    def init_weights(self):
        """
        Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        The weights_init function takes an initialized model as input and reinitializes all convolutional,
        convolutional-transpose, and batch normalization layers to meet this criteria. 
        This function is applied to the models immediately after initialization. 

        params : None

        Returns : None
        """
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
          module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
          nn.init.normal_(m.weight.data, 1.0, 0.02)
          nn.init.constant_(m.bias.data, 0)
            
    def forward(self, x):
        """
        Forward function
        Passes the layers to the activation function ( Relu was primarily used)
        Pooling is done 
        Dropout is used 
        Last layer doesn't use an activation function 

        Parameters : x

        Return : x
        """

        # CNN + Relu + Dropout
        x = self.conv1(x)
        x = self.Dropout(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        x = self.Dropout(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.Dropout(x)
        x = F.relu(self.conv6(x))
        x = self.Dropout(x)
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        return x
        

class Discriminator(nn.Module):
    """
    The discriminator is a binary classification network that takes an image as input 
    and outputs a scalar probability that the input image is real (as opposed to fake).
    This class create a basic CNN with 4 CNN Layers and 5 Fully connected layers
    The output is a binary classification problem 

    Functions : forward, get_loss , get_optimizer, weight_init
    """
    class Params:
        """
        Inherited Class
        Calls the OptimizerParams class and initalizes an instance of it

        Functions : None

        """
        opt = OptimizerParams()

    def __init__(self, params, n_channels=3, img_size=32):
        """
        Constructor Class 
        The constructor initializes the model with the required CNN layers
        Definition of the CNN Net we are using and all the layers
        Batch Normalization was used for most of the layers
        Dropout was also used to prevent Overfitting
        All the CNN layers are weight initalized by the init_weights function
        As this is the discriminator class it is a binary classification problem
        
        params : params, n_channels, img_size

        returns : None

        """
        super(Discriminator, self).__init__()

        # CNN + Batch Normalization 
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(img_size*img_size*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,32)
        self.fc5 = nn.Linear(32,1)

        # Dropout + Sigmoid function used to squash values between 0 and 1
        self.Dropout = nn.Dropout(0.2)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """
        Forward function
        Passes the layers to the activation function ( Relu was primarily used)
        Pooling is done 
        Dropout is used 
        Last layer uses the activation function sigmoid
        Before passing x into the FCC we resize it 

        Parameters : x

        Return : x
        """

        # CNN + Relu + Dropout
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) 
        x = self.conv4(x)
        x = self.Dropout(x)

        # Resize for FCC
        x = x.view(x.size(0),3*32*32)

        # FCC + Relu
        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.Dropout(x)
        x = self.fc5(x)
        return self.out_act(x)
        
    def get_loss(self, dscr_out, labels):
        """
        Reference : https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        This function uses the BCE loss function and returns the error.
        Labels.unsqueeze(1) was used to increase the add a dimension to labels 
        This was done so that it matches the documentation so that it matches the target dimension 
        Then it was converted to float values otherwise a CUDA error was raised
            1 - Input: (N, *)where *∗ means, any number of additional dimensions
            2 - Target: (N, *), same shape as the input
            3 -  Output: scalar. If reduction is 'none', then (N, *), same shape as input.

        :param dscr_out: Discriminator output
        :param labels: Real vs fake binary labels (real --> 1, fake --> 0)
        :return: error
        """
        criterion = nn.BCELoss()
        labels = labels.unsqueeze(1)
        labels = labels.float()
        error = 0
        error = criterion(dscr_out,labels)
        return error

    def get_optimizer(self, modules):
        """
        This function returns the optimizer ie Adam with the parameters set
        An instance of the Optimizerparameter class is called

        :param nn.ModuleList modules: [shared_net, discriminator, classifier, generator, composite_loss]
        :return: torch.optim.Adam
        """
        opt = OptimizerParams()
        return torch.optim.Adam(Generator.parameters(self), lr=opt.lr, weight_decay = opt.weight_decay)

    def init_weights(self):
        """
        Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        The weights_init function takes an initialized model as input and reinitializes all convolutional,
        convolutional-transpose, and batch normalization layers to meet this criteria. 
        This function is applied to the models immediately after initialization. 

        params : None

        Returns : None
        """
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
          module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
          module.weight.data.normal_(1.0, 0.02)
          module.bias.data.fill_(0)
            

class Classifier(nn.Module):
    """

    This class deals with the classification of the images
    It is a multi classification model 
    Uses the knowledge from the shared net class and tries to classify the images

    Functions : forward, init_weight, loss
    """
    class Params:
        """
        Inherited Class
        Dummy variable set to 0 as it inherits no Parameters

        Functions : None

        """
        dummy = 0

    def __init__(self, params, n_classes=20, n_channels=3, img_size=32):
        """
        Constructor Class 
        The constructor initializes the model with the required CNN layers
        Definition of the CNN Net we are using and all the layers
        Batch Normalization was used for most of the layers
        Dropout was also used to prevent Overfitting
        All the CNN layers are weight initalized by the init_weights function
        As this is the Classifier class it is a Multi classification problem
        Fully connected layers were also used

        :param n_classes: 10 classes for real images and 10 for fake images
        :param Classifier.Params params, n_classes, n_channels , img_size

        Returns : None
        """
        super(Classifier, self).__init__()
        
        # CNN + Batch Normalization 
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 3,kernel_size = 3,stride = 1,padding = 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(img_size*img_size*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,20)

        # Dropout
        self.Dropout = nn.Dropout(0.2)


    def get_loss(self, cls_out, labels, is_labeled=None, n_labeled=None):
        """
        Reference : https://pytorch.org/docs/stable/tensors.html
        This function returns the Cross entropy loss for the classifier model
        A list is first initalized which keeps track of all the valid labels 
        from the is_labeled list

        If is_labeled is None it implies all of them are valid then just pass 
        the same classifier output(cls_out) and labels to the criterion
        
        If not then we iterate over the length of the is_labeled tensor and
        Whenever we encounter a valid label( True) we append 1 else 0

        Then we use torch.ByteTensor(indices) which keeps track of all the valid labels
        Using this tensor we extract the valid rows from cls_out and labels
        Then these both are passed to the criterion and the loss is returned

        :param cls_out: Classifier output
        :param labels: labels for both fake and real images in range 0 - 19;
        -1 for real unlabeled, -2 for fake unlabeled
        :param is_labeled: boolean array marking which labels are valid; None when all are valid
        :param n_labeled: number of valid labels;  None when all are valid
        
        Return: Loss
        """
        criterion = nn.CrossEntropyLoss()
        indices = []
        if is_labeled == None:
          loss = 0
          loss = criterion(cls_out,labels)
          return loss
        else:
          for i in range(len(is_labeled)):
            if is_labeled[i]:
              indices.append(1)
            else:
              indices.append(0)
          extract = torch.ByteTensor(indices)
          cls_out = cls_out[extract]
          labels = labels[extract]
          loss = 0
          loss = criterion(cls_out,labels)
          return loss

    def init_weights(self):
        """
        Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        The weights_init function takes an initialized model as input and reinitializes all convolutional,
        convolutional-transpose, and batch normalization layers to meet this criteria. 
        This function is applied to the models immediately after initialization. 

        params : None

        Returns : None
        """
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
          module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
          module.weight.data.normal_(1.0, 0.02)
          module.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward function
        Passes the layers to the activation function ( Relu was primarily used)
        Pooling is done 
        Dropout is used 
        Last layer uses Softmax
        Before passing x into the FCC we resize it

        Parameters : x

        Return : x
        """

        # CNN + Relu + Dropout
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) 
        x = self.conv4(x)
        x = self.Dropout(x)
        x = self.conv5(x)

        # Resize for FCC
        x = x.view(x.size(0),3*32*32)

        # FCC + Relu
        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.Dropout(x)
        x = F.relu(self.fc4(x))
        x = self.Dropout(x)
        x = self.fc5(x)
        return F.softmax(x,dim = 1)


class Generator(nn.Module):
    """
    Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    The generator, G, is designed to map the latent space vector (z) to data-space.
    Since our data are images, converting z to data-space means ultimately 
    creating a RGB image.
    The generator has to be good at creating fake images to fool the discriminator

    Functions : forward, init_weight,get_optimizer
    """
    class Params:
        """
        Inherited Class
        Initalizes the Optimizer Parameters class and gets an instance of it 

        Functions : None

        """
        opt = OptimizerParams()

    def __init__(self, params, input_size, n_channels=3, out_size=32):
        """
        Constructor Class 
        The constructor initializes the model with the required CNN transpose layers
        Definition of the CNN  Net we are using and all the layers
        Batch Normalization was used for most of the layers
        Dropout was also used to prevent Overfitting
        All the CNN layers are weight initalized by the init_weights function
        Finally Tanh was used as referered by the reference material as well as the paper
        
        params : params, n_channels, img_size

        returns : None

        """
        super(Generator, self).__init__()
        # CNN transpose + Batch Normalization 
        self.conv1t = nn.ConvTranspose2d(input_size, 256, 4, 1, 0)
        self.conv1t_bn = nn.BatchNorm2d(256)
        self.conv2t = nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv2t_bn = nn.BatchNorm2d(128)
        self.conv3t = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv3t_bn = nn.BatchNorm2d(64)
        self.conv4t = nn.ConvTranspose2d(in_channels = 32, out_channels = n_channels, kernel_size = 4, stride = 2, padding = 1)

        # Dropout Layer + Tanh activation function
        self.Dropout = nn.Dropout(0.2)
        self.act = nn.Tanh()
        
    def forward(self, x):
        """
        Forward function
        Passes the layers to the activation function ( Relu was primarily used)
        Pooling is done 
        Dropout is used 
        Last layer uses an activation function 

        Parameters : x

        Return : x
        """
        x = self.conv1t(x)
        x = F.relu(self.conv2t(x))
        x = self.Dropout(x)
        x = F.relu(self.conv3t(x))
        x = self.Dropout(x)
        x = F.relu(self.conv4t(x))
        x = self.act(x)
        return x

        
    def get_optimizer(self, module):
        """
        This function returns the optimizer ie Adam with the parameters set
        An instance of the Optimizerparameter class is called

        :param nn.ModuleList modules: [shared_net, discriminator, classifier, generator, composite_loss]
        :return:
        """
        opt = OptimizerParams()
        return torch.optim.Adam(Generator.parameters(self), lr=opt.lr, weight_decay = opt.weight_decay) 

    def init_weights(self):
        """
        Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        The weights_init function takes an initialized model as input and reinitializes all convolutional,
        convolutional-transpose, and batch normalization layers to meet this criteria. 
        This function is applied to the models immediately after initialization. 

        params : None

        Returns : None
        """
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
          module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
          module.weight.data.normal_(1.0, 0.02)
          module.bias.data.fill_(0)


class CompositeLoss(nn.Module):
    """
    This class deals with multiple Loss functions
    CompositeLoss takes the classifier and discriminator losses as input and combines them to produce the overall loss that
    will be used to train them.

    Functions : forward

    """
    class Params:
        """
        Inherited Class
        Dummy variable set to 0 as it inherits no Parameters

        Functions : None

        """
        dummy = 0

    def __init__(self, device, params):
        """
        Constructor class 
        Nothing to initalize apart from device

        :param torch.device device:
        :param CompositeLoss.Params params:
        """
        super(CompositeLoss, self).__init__()
        self.device = device

    def forward(self, dscr_loss, cls_loss):
        """
        This function calculates the forward pass and returns the weighted sum 
        of the losses by the BCE AND Crossentropy loss
        a = 0.1 was used as a weight
        params : dscr_loss, cls_loss

        return : error
        """
        a = 0.1
        error = 0
        error = (a*dscr_loss + (1-a)*cls_loss)
        return error

class SaveCriteria:
    """
    This Class deals with when to save the model with the required values
    A simple example has already been given
    No modifications were made 

    functions : decide
    """
    def __init__(self, status_df):
        """

        :param pd.DataFrame status_df:
        """
        self._opt_status_df = status_df.copy()

    def decide(self, status_df):
        """
        decide when to save new checkpoint while training based on training and validation stats
        following metrics are available:
      |   dscr_real |   cls_real |   cmp_real |   dscr_fake |   cls_fake |   cmp_fake |   cmp |   dscr_gen |   cls_gen |
         cmp_gen |   total_acc |   real_acc |   fake_acc |   fid |   is |

         where the first 10 are losses: dscr --> discriminator, cls --> classifier, gen --> generator,
         cmp --> composite, real --> real images, fake --> fake images

         acc --> classification accuracy
         is --> inception_score

        :param pd.DataFrame status_df:
        """

        save_weights = 0
        criterion = ''

        """total train accuracy over real+fake images"""
        if status_df['total_acc']['valid'] > self._opt_status_df['total_acc']['valid']:
            self._opt_status_df['total_acc']['valid'] = status_df['total_acc']['valid']
            save_weights = 1
            criterion = 'valid_acc'

        if status_df['total_acc']['train'] > self._opt_status_df['total_acc']['train']:
            self._opt_status_df['total_acc']['train'] = status_df['total_acc']['train']
            save_weights = 1
            criterion = 'train_acc'

        """composite loss on real images"""
        if status_df['cmp_real']['valid'] > self._opt_status_df['cmp_real']['valid']:
            self._opt_status_df['cmp_real']['valid'] = status_df['cmp_real']['valid']
            save_weights = 1
            criterion = 'valid_loss'

        if status_df['cmp_real']['train'] > self._opt_status_df['cmp_real']['train']:
            self._opt_status_df['cmp_real']['train'] = status_df['cmp_real']['train']
            save_weights = 1
            criterion = 'train_loss'

        return save_weights, criterion
