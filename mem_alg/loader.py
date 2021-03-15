from keras.datasets import mnist

def load():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    """
    print('X_train: ' + str(train_x.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_x.shape))
    print('Y_test:  ' + str(test_y.shape))
    """
    train = []
    test = []
    for i in range(10):
        train.append([])
        test.append([])

    for i in range(60000):
        train[train_y[i]].append(train_x[i])

    for i in range(10000):
        test[test_y[i]].append(test_x[i])

    """
    print("train:")
    for i in range(10):
        print(f"{i}: ({len(train[i])})")
        
    print()
    
    print("test:")
    for i in range(10):
        print(f"{i}: ({len(test[i])})")
    """
    return train, test
