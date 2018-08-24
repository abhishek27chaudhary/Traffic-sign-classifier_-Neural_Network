import glob
import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import sys
sys.path.append('/home/krishna_warrior/Desktop/traffic_sign_final')        #address for pipeline folder
from pipeline import NeuralNetwork, make_adam, Session, build_pipeline      #import from pipeline
matplotlib.style.use('ggplot')

TRAIN_IMAGE_DIR='/home/krishna_warrior/Desktop/dataset30k'
dfs=[]
for train_file in glob.glob(os.path.join(TRAIN_IMAGE_DIR,'*/GT-*.csv')):
    folder=train_file.split('/')[5]      #actually my path contains 5 elements, configure according to your path
    df=pd.read_csv(train_file,sep=';')
    df['Filename']=df['Filename'].apply(lambda x: os.path.join(TRAIN_IMAGE_DIR,folder,x))
    dfs.append(df)

train_df=pd.concat(dfs,ignore_index=True)             #storing csv data in train_df
train_df.head();

#print(train_df['ClassId'])
#print(dfs)

df=pd.DataFrame(train_df)
df.to_csv("my_data1.csv",index=False)

#y= train_df['ClassId'][~np.isnan(train_df['ClassId'])]      #this is used to remove nan entries

df=pd.DataFrame(train_df['ClassId'])
df.to_csv("my_data.csv",index=False)

N_CLASSES=np.unique(train_df['ClassId']).size   #no of classes
print("No. of training images: {:>5}".format(train_df.shape[0]))
print("No. of classes: {:>5}".format(N_CLASSES))

def class_dist(classIDs,title):            #maping data distribution with help  of matplotlib
    plt.figure(figsize=(15,5))
    plt.title('class id dist for {}'.format(title))
    plt.hist(classIDs,bins=N_CLASSES)
    plt.show()

#class_dist(train_df['ClassId'],'Train data')
sign_name_df=pd.read_csv('sign_names.csv',index_col='ClassId')       #getting sign name corresponding to classid
sign_name_df.head()

#print(sign_name_df)
sign_name_df['Occurence']=[sum(train_df['ClassId']==c) for c in range(N_CLASSES)]      # add new colum 'Occurence' which stores no of images in a class.

sign_name_df.sort_values('Occurence',ascending=False)    #sort occurence in non ascending order
#print(sign_name_df)                       #will show id,name,occurence of images.
SIGN_NAMES=sign_name_df.SignName.values         #store sign names value corresponding to id
#print(SIGN_NAMES[2])      #will  print speed limit 50 km/h

def load_image(image_file):
    return plt.imread(image_file)
def get_samples(image_data,num_samples,class_id=None):         #prepare sample images(VALIDATION)
    if class_id is not None:
        image_data=image_data[image_data['ClassId']==class_id]
    indices=np.random.choice(image_data.shape[0],size=num_samples,replace=False)   #randomly chosing images for validation
    return image_data.iloc[indices][['Filename','ClassId']].values

def show_images(image_data,cols=5,sign_names=None,show_shape=False,func=None):        #displey all images in matrix form
    num_images=len(image_data)
    rows=num_images//cols
    plt.figure(figsize=(cols*3,rows*2.5))
    for i,(image_file,lable) in enumerate(image_data):
        image=load_image(image_file)
        if func is not None:
            image=func(image)
        plt.subplot(rows,cols,i+1)
        plt.imshow(image)
        if sign_names is not None:
            plt.text(0,0,'{}:{}'.format(lable,sign_names[lable]),color='k',backgroundcolor='c',fontsize=8)
        if show_shape:
            plt.text(0,image.shape[0],'{}'.format(image.shape),color='k',backgroundcolor='y',fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()

sample_data=get_samples(train_df,20)

#show_images(sample_data,sign_names=SIGN_NAMES,show_shape=True)
#show_images(get_samples(train_df,40,class_id=2),cols=20,show_shape=True)
#-------------------------------------------------------------------------------------------------------------------
#  Train and validation
X=train_df['Filename'].values
y=train_df['ClassId'].values
print('X data',len(X))
X_train, X_valid,y_train, y_valid=train_test_split(X,y,stratify=y,test_size=329,random_state=0)      #sorry aashita :) i will use external validation images later
print('X_train:',len(X_train))
print('X_valid:',len(X_valid))
#Model implimentation
INPUT_SHAPE=(32,32,3)

#pipeline
def train_evaluate(pipeline,epochs=10,samples_per_epoch=50000,train=(X_train,y_train),test=(X_valid, y_valid)):     #here 50000 is used to increase accuracy(more iterations)
    X,y=train
    learning_curve=[]
    for i in range (epochs):
        indices=np.random.choice(len(X),size=samples_per_epoch)       #did u get the use of np.random() ?
        pipeline.fit(X[indices],y[indices])
        scores=[pipeline.score(*train),pipeline.score(*test)]
        learning_curve.append({i, *scores})
        print("Epoch: {:>3} Training score: {:.3f} Evaluation score: {:.3f}".format(i,*scores))
    return np.array(learning_curve).T

#network1 performance:
def resize_image(image,shape=INPUT_SHAPE[:2]):
    return cv2.resize(image,shape)
loader=lambda image_file: resize_image(load_image(image_file))
'''with Session() as session:
    functions=[loader]
    pipeline=build_pipeline(functions,session, network1(),make_adam(1.0e-3))
    train_evaluate(pipeline)'''

#image aug
def rBrightness(image,ratio):
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness=np.float64(hsv[:, :, 2])
    brightness=brightness*(1.0+np.random.uniform(-ratio,ratio))
    brightness[brightness>255]=255
    brightness[brightness<0]=0
    hsv[:, :, 2]=brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def rRotation(image, angle):
    if angle==0:
        return image
    angle=np.random.uniform(-angle,angle)
    rows,cols=image.shape[:2]
    size=rows,cols
    center=rows/2,cols/2
    scale=1.0
    rotation=cv2.getRotationMatrix2D(center,angle,scale)
    return cv2.warpAffine(image,rotation,size)

def rTranslation(image, translation):
    if translation==0:
        return 0
    rows,cols=image.shape[:2]
    size=rows,cols
    x=np.random.uniform(-translation,translation)
    y=np.random.uniform(-translation,translation)
    trans=np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image,trans,size)

def rShear(image, shear):
    if shear==0:
        return image
    rows,cols=image.shape[:2]
    size=rows,cols
    left,right,top,bottom=shear,cols-shear,shear,rows-shear
    dx=np.random.uniform(-shear,shear)
    dy=np.random.uniform(-shear,shear)
    p1=np.float32([[left,top],[right,top],[left,bottom]])
    p2=np.float32([[left+dx,top],[right+dx,top+dy],[left,bottom+dy]])
    move=cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image,move,size)

def agument_image(image,brightness,angle,translation,shear):
    image=rBrightness(image,brightness)
    image=rRotation(image,angle)
    image=rTranslation(image,translation)
    image=rShear(image,shear)
    return image

augmenter=lambda x: agument_image(x,0.7,10,5,2)
show_images(sample_data[10:],cols=10)

for i in range(5):
    show_images(sample_data[10:],cols=10,func=augmenter)

'''with Session() as session:
    functions=[loader,augmenter]
    pipeline=build_pipeline(functions,session, network1(),make_adam(1.0e-3))
    train_evaluate(pipeline)'''

normilzers=[('x-127.5', lambda x: x-127.5),('x/127.5-1.0', lambda x: x/127.5-1.0),
           ('x/225.0-0.5',lambda x: x/225.0-0.5),('x-x.mean()',lambda x: x-x.mean()),
           ('(x-x.mean())/x.std()', lambda x: (x-x.mean())/x.std())]

normalizer=lambda x: (x-x.mean())/x.std()

converters = [('Gray', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]),
              ('HSV', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)),
              ('HLS', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HLS)),
              ('Lab', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2Lab)),
              ('Luv', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2Luv)),
              ('XYZ', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2XYZ)),
              ('Yrb', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)),
              ('YUV', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV))]

GRAY_INPUT_SHAPE = (*INPUT_SHAPE[:2], 1)
preprocessors = [loader,augmenter, normalizer]

def show_learning_curve(learning_curve):
    epochs, train, valid = learning_curve
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, train, label='train')
    plt.plot(epochs, valid, label='validation')
    plt.title('Learning Curve')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.xticks(epochs)
    plt.legend(loc='center right')

def plot_confusion_matrix(cm):
    cm = [row/sum(row)   for row in cm]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Oranges)
    fig.colorbar(cax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class IDs')
    plt.ylabel('True Class IDs')
    plt.show()

def print_confusion_matrix(cm, sign_names=SIGN_NAMES):
    results = [(i, SIGN_NAMES[i], row[i]/sum(row)*100) for i, row in enumerate(cm)]
    accuracies = []
    for result in sorted(results, key=lambda x: -x[2]):
        print('{:>2} {:<50} {:6.2f}% {:>4}'.format(*result, sum(y_train==result[0])))
        accuracies.append(result[2])
    print('-'*50)
    print('Accuracy: Mean: {:.3f} Std: {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))

def make_network3(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24]) # <== doubled
            .max_pool()
            .relu()
            .conv([5, 5, 64]) # <== doubled
            .max_pool()
            .relu()
            .flatten()
            .dense(480)  # <== doubled
            .relu()
            .dense(N_CLASSES))

X_new = np.array(glob.glob('images/*.ppm'))
new_images = [plt.imread(path) for path in X_new]

print('-' * 80)
print('New Images for Random Testing')
print('-' * 80)

plt.figure(figsize=(15,5))
for i, image in enumerate(new_images):
    plt.subplot(2,len(X_new)//2,i+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()

print('getting top 5 results')

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network3_e-100_lr-1.0e-4.ckpt')
    prob = pipeline.predict_proba(X_new)
    estimator = pipeline.steps[-1][1]
    top_5_prob, top_5_pred = estimator.top_k_

print('done')
print('-' * 80)
print('Top 5 Predictions')
print('-' * 80)

for i, (preds, probs, image) in enumerate(zip(top_5_pred, top_5_prob, new_images)):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    for pred, prob in zip(preds.astype(int), probs):
        sign_name = SIGN_NAMES[pred]
        print('{:>5}: {:<50} ({:>14.10f}%)'.format(pred, sign_name, prob*100.0))
    print('-' * 80)