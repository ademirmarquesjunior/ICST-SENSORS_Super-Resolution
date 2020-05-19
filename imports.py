import tensorflow as tf
import numpy as np
import csv
from PIL import Image, ImageChops, ImageStat
import sewar as sw
import rasterio as rt
import cv2

#from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import os

#Importando bibliotecas do Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.model_selection import cross_val_score


#Importando a biblioteca do Keras para redes neurais
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
#from keras.callbacks import TensorBoard

#from keras.models import model_from_json