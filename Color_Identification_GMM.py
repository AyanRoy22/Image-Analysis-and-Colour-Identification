from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76




def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    df = pd.DataFrame(modified_image, columns=['Red','Green','Blue'])
    df['Pixels'] = df.index
    
    gmm = GaussianMixture(n_components = 5)
    gmm.fit(df)

    labels = gmm.predict(df)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    df['labels'] = labels
    d0 = df[df['labels']==0]
    d1 = df[df['labels']==1]
    d2 = df[df['labels']==2]
    d3 = df[df['labels']==3]
    d4 = df[df['labels']==4]



    d = [d0, d1, d2, d3, d4]
    list = []
    for i in d:
        cr = int(i['Red'].mean())
        cg = int(i['Green'].mean())
        cb = int(i['Blue'].mean())
        
        rgb_colors = (cr, cg, cb)
        hex_colors = ('#%02x%02x%02x' % (cr, cg, cb))
        list.append(hex_colors)

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = list, colors = list)
    
    return list



image = cv2.imread('picture1.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


get_colors(get_image('picture1.jpg'), 5, True)