# Control of car using hand gestures

## Table of contents
- [Team Members](#team-members)
- [Project Demo](#project-demo)
- [Desricption](#description)
- [Technologies](#technologies)
- [Project Video](#project-video)
- [Contributing](#contributing)






## Team Members
- Omar Gamal
- Karim Ibrahim Yehia
- Eslam Mohamed Abdelbassir
- Ahmed Hesham
- Karim Kamal Rizk
- Ahmed Samir



## Project Demo
This project we have designed a simple Hand Gesture Controlled Robot using RasperryPi to classify 4 hand gestures (pointer,stop,punch,peace) using ensemble of deep learning architectures for control of car robot.This Hand Gesture Controlled Robot is based on RasperryPi, WebCam and L293D Motor Driver. 


## Desricption

# coding: utf-8

# ## Load data

# In[1]:


import h5py
import numpy as np

def load_dataset():
    train_dataset = h5py.File('data/train_signs.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) # your train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:]) # your train set labels

    test_dataset = h5py.File('data/test_signs.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:]) # your test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:]) # your test set labels

    classes = np.array(test_dataset['list_classes'][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

X_train, y_train, X_val, y_val, classes = load_dataset()


# Visualize a sample data

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

index = 0
plt.imshow(X_train[index])
plt.title('Label {}'.format(np.squeeze(y_train[:, index])))

plt.show()


# Examine shapes of data

# In[3]:


print("Train data:", X_train.shape)
print("Train labels:", y_train.shape)

print("Validation data:", X_val.shape)
print("Validation labels:", y_val.shape)


# ## Preprocess data

# We can see that data is composed of [m, H, W, C], which:
# + m: Number of data
# + H, W: Picture's height & width
# + C: Number of channels
#     
# For Pytorch, we need [m, C, H, W]

# In[4]:


X_train = X_train.reshape(-1, 3, 64, 64)
X_val = X_val.reshape(-1, 3, 64, 64)


# Normalization: Devide by 255.

# In[5]:


X_train = X_train / 255.
X_val = X_val / 255.


# ## Make dataset/dataloader

# Convert data to Pytorch's tensors

# In[6]:


import torch

X_train_tensor = torch.from_numpy(X_train.astype('float32'))
y_train_tensor = torch.from_numpy(np.squeeze(y_train).astype('int'))

X_val_tensor = torch.from_numpy(X_val.astype('float32'))
y_val_tensor = torch.from_numpy(np.squeeze(y_val).astype('int'))


# Make dataset from tensors

# In[7]:


from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


# Test dataset maker

# In[8]:


print("Length of dataset:", len(train_dataset))

X_sample, y_sample = train_dataset[0]
print("Sample X:", X_sample)
print("Sample y:", y_sample)


# Now make a data loader

# In[9]:


from torch.utils.data import DataLoader

BATCH_SIZE = 128

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)


# ## Build model

# A basic CNN

# In[10]:


import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*13*13, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, len(classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # flatten [m, C, H, W] -> [m, C*H*W]
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# In[11]:


model = CNN()
model = model.cuda() # use GPU
print(model)


# In[12]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ## Train

# In[13]:


from torch.autograd import Variable

def fit(epochs):
    history = {
        'loss': [],
        'val_acc': []
    }
    
    def train(epoch):
        model.train()

        ITERATIONS = len(train_dataset) // BATCH_SIZE + 1
        for i, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda() # use GPU

            ## forward 
            output = model(data)

            ## compute loss
            loss = criterion(output, target)

            ## bacwward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## Regularly check
            if (i+1) % 3 == 0:
                print("({}) [{}/{}] loss: {:.4f}".format(epoch+1, i+1, ITERATIONS,
                                                         loss.data[0]))

        ## save training loss
        history['loss'].append(loss.data[0])


    def validate():
        model.eval()

        total = len(val_dataset)
        correct = 0
        for data, target in val_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            data, target = data.cuda(), target.cuda() # use GPU

            # forward
            output = model(data)

            # get index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum() # check equality, then sum up

        val_acc = correct/total
        print("val_acc: {:.2f}".format(correct/total))
        
        # Save validation accuracy
        history['val_acc'].append(val_acc)        

    ## Start training, validate accuracy each epoch
    for epoch in range(0, epochs):
        train(epoch)
        validate()
    
    return history


# In[14]:


EPOCHS = 30
history = fit(EPOCHS)


# Visualize training process

# In[15]:


epochs = range(1, EPOCHS+1)

plt.plot(epochs, history['loss'], 'r')
plt.title('Training loss')
plt.show()


# In[16]:


plt.plot(epochs, history['val_acc'], 'b')
plt.title('Validation accuracy')
plt.show()


# ## Test

# In[17]:


def predict(X):
    """Arguments
        X: numpy image, shape (H, W, C)
    """
    # reshape
    shape = X.shape
    X = X.reshape((1, shape[2], shape[0], shape[1]))
    # normalize
    X = X / 255.
    
    X = torch.from_numpy(X.astype('float32'))
    X = Variable(X, volatile=True)
    
    # forward
    output = model.cpu()(X)

    # get index of the max
    _, index = output.data.max(1, keepdim=True)
    return classes[index[0][0]] # index is a LongTensor -> need to get int data


# Sample predict from validation data

# In[18]:


import random

_, _, X_val, y_val, classes = load_dataset()
index = random.randint(0, len(X_val))

pred = predict(X_val[index])

plt.imshow(X_val[index])
plt.title('Label: {}, Predict: {}'.format(np.squeeze(y_val[:,index]), pred))
plt.show()


# ## Export

# Extract model's trained weights

# In[20]:


import os

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'cnn.pth')

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
torch.save(model.state_dict(), MODEL_PATH)



## Technologies
created with:



## Project Video
[Youtube](https://youtu.be/YxlOvFcngS4)





## Contributing
The preprocessing and collection of data is based on the following tutorial [ Computer Vision and Machine Learning with Python, Keras and OpenCV](https://github.com/jrobchin/Computer-Vision-Basics-with-Python-Keras-and-OpenCV/).


