import os 
import numpy as np 
import pandas as pd
from PIL import Image 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import matplotlib.pyplot as plt

def list_dataset(dir_source):
    
    pos = os.listdir(os.path.join(dir_source,'COVID'))
    neg = os.listdir(os.path.join(dir_source,'non-COVID'))
    
    y = [1] * len(pos) + [0] * len(neg)
    X_path = [os.path.join(dir_source,'COVID', x) for x in pos] + [os.path.join(dir_source,'non-COVID', x) for x in neg]
    
    img_ref = pd.DataFrame({'x_path': X_path, 'y': y})

    return img_ref


def prepare_img(img, new_W = None, new_H = None, gray = True):
    
    if (new_W is not None) and (new_H is not None):
        img = img.resize((new_W, new_H), Image.ANTIALIAS)
    if gray:
        img = img.convert('LA')
        img_np = np.array(img)
        if img_np.shape[2] == 2:
            img = img_np[:,:,0]
            
    return img



def flatten_imgs(img_paths, new_W = None, new_H = None, gray = True):
    
    flattened_imgs = []
    
    for img_path in img_paths:
        img = Image.open(img_path)
        img = prepare_img(img, new_W = new_W, new_H = new_H, gray = gray)
        flattened_imgs.append(img.flatten())
        
    return flattened_imgs

def make_dataset(img_ref, new_W = None, new_H = None, gray = True):
    
    img_paths = list(img_ref['x_path'])
    
    flattened_imgs = flatten_imgs(img_paths, new_W = new_W, new_H = new_H, gray = gray)
    
    X = np.array(flattened_imgs)
    X = pd.DataFrame(X)

    y = list(img_ref['y'])
    
    return X,y



dir_source = 'data'
img_ref = list_dataset(dir_source)
X, y = make_dataset(img_ref, new_W = 100, new_H = 100, gray = True)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

lr = LogisticRegression()
lr.fit(X_train, y_train)


y_hat = lr.predict(X_test)

acc = accuracy_score(y_test, y_hat)
precision = precision_score(y_test, y_hat) 
recall = recall_score(y_test, y_hat)
f1 = f1_score(y_test, y_hat)