def knn(x_train, y_train, x_test, n_classes, device,k):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    """
    create a numpy array of zeros for predictions of size 1000 x 1
    dtype is automatically recognized by analysing y_train for classification purpose
    """
    ypred = np.zeros(1000,dtype = y_train.dtype)

    """ 
    Convert training and test arrays to tensor ; reference = https://pytorch.org/docs/stable/tensors.html
    dtypes are set to float as well as the device is automatically recognized
    """
    x_train= torch.tensor(x_train, dtype=torch.float, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float, device=device)

    """ 
    loop over the range of 1000 testing images
    """
    for index in range(x_test.shape[0]):

      """ 
      find the L2 norm of the training set - test set (here p = 2 means the L2 norm)
      After hyper paramenter tuning the best value of k was found to be 1
      the p value was found to be 4 which yielded a best accuracy of 97% although slower which yielded in a loss of 2 seconds
      the p value = 2 that is the euclidean distance yielded 96.3% in 4.4 at best and on average at 5 seconds.
      """
      distances = torch.norm(x_train-x_test[index],p = 4,dim = 1)

      """
      find the min element and the index it is at ; reference = https://pytorch.org/docs/stable/generated/torch.topk.html
      largest is set to false as we want the smallest element and not the largest one
      """
      element,min_index = torch.topk(distances,1,largest = False)
  
      """
      find the value of it in the y_train set and assign it to predictions
      """
      ypred[index] = y_train[min_index]
      
    return ypred