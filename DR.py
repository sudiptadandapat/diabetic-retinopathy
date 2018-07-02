from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import pywt
import os




img_rows=img_cols=200
immatrix=[]
im_unpre = []

dim=(768,576)

for i in range(1,90):
    img_pt =''
    if i < 59:
        img_pt = img_pt + "normal" + str(i) + ".jpg"
    else:
        img_pt = img_pt + "severe" + str(i-58)+ ".jpg"

    print(img_pt)
    img = cv2.imread(img_pt)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)
    immatrix.append(np.array(equ).flatten())

cv2.imshow("equ2",equ)
cv2.waitKey(0)    



imm_dwt = []
for equ in immatrix:
    equ = equ.reshape((576,768))
    coeffs = pywt.dwt2(equ, 'haar')
    equ2 = pywt.idwt2(coeffs, 'haar')
    imm_dwt.append(np.array(equ2).flatten())


def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)

    
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    
    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

   
    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    # return the "convolution" kernel for filter2D
    return cv2.flip(kernel, -1) 





def show_images(images,titles=None, scale=1.3):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()





def gaussian_matched_filter_kernel(L, sigma, t = 3):
    '''
    K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, True)




def createMatchedFilterBank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

bank_gf = createMatchedFilterBank()
imm_gauss2 = []
for equ2 in imm_dwt:
    equ2 = equ2.reshape((576,768))
    equ3 = applyFilters(equ2,bank_gf)
    imm_gauss2.append(np.array(equ3).flatten())



e_ = equ3
np.shape(e_)
e_=e_.reshape((-1,3))
print(np.shape(e_))


imm_kmean = []
for equ3 in imm_gauss2:
    img = equ3.reshape((576,768))
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    k=cv2.KMEANS_PP_CENTERS


    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res3=cv2.subtract(255,res2)
    imm_kmean.append(np.array(res3).flatten())

cv2.imshow("k1",res2)
cv2.waitKey(0)
cv2.imwrite("k1.jpg",res2)
cv2.imshow("k2",res3)
cv2.waitKey(0)
cv2.imwrite("k2.jpg",res3)


# the array ranges from 0 - 89
np.shape(imm_kmean)
plt.imshow(imm_kmean[76].reshape((576,768)),cmap="gray")
plt.show()



from sklearn.svm import SVC
clf = SVC()

Y = np.ones(89)

for i in range(0,59):
    Y[i]=0


print(clf.fit(imm_kmean, Y))

y_pred = clf.predict(imm_kmean)

k=[1,4,7,9,12,15,16,17,18,19,20,21,22,27,29,32,35,36,37,38,39,40,42,43,46,48,49,52,53,54,55,56,57,58,59,63,64,65,66,67,68,69,70,74,76,77,79,81,84,86,87,89]
k = k-np.ones(len(k))
print(k)

k =[int(x) for x in k]

imm_train = []
y_train = []
for i in k:
    imm_train.append(imm_kmean[i])
    y_train.append(Y[i])



clf.fit(imm_train, y_train)

y_pred = clf.predict(imm_kmean)



from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix:')
cm1=confusion_matrix(Y,y_pred)
print(cm1)
print('\n')
print(classification_report(Y,y_pred))


print("TP =",cm1[0,0],"FP =",cm1[0,1])
print("FN =",cm1[1,0],"TN =",cm1[1,1])
tp=cm1[0,0]
fp=cm1[0,1]
fn=cm1[1,0]
tn=cm1[1,1]

print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
print("Sensitivity = ",float(tp/(tp+fn+1)))
print("Specificity = ",float(tn/(tn+fp+1)))

print("PPV = ",float(tp/(tp+fp+1)))



from sklearn.metrics import mean_absolute_error
print('\n Mean Absolute Error=',mean_absolute_error(Y, y_pred))
def rmse(y_pred, y_roc):
	return np.sqrt(((y_pred - y_roc) ** 2).mean())
y_roc = np.array(Y)
rms=rmse(y_pred,y_roc)
print('\n Root Mean Square Error=',rms)
import pylab as pl
from sklearn.metrics import roc_curve, auc
y_roc = np.array(Y)
fpr, tpr, thresholds = roc_curve(y_roc, y_pred)
roc_auc = auc(fpr, tpr)
print("\n Area under the ROC curve : %f" % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.show()



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(imm_train, y_train)

y_pred2=neigh.predict(imm_kmean)


from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix:')
cm2=confusion_matrix(Y,y_pred2)
print(cm2)
print('\n')
print(classification_report(Y,y_pred2))

print("TP =",cm2[0,0],"FP =",cm2[0,1])
print("FN =",cm2[1,0],"TN =",cm2[1,1])
tp=cm2[0,0]
fp=cm2[0,1]
fn=cm2[1,0]
tn=cm2[1,1]

print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
print("Sensitivity = ",float(tp/(tp+fn+1)))
print("Specificity = ",float(tn/(tn+fp+1)))

print("PPV = ",float(tp/(tp+fp+1)))



from sklearn.metrics import mean_absolute_error
print('\n Mean Absolute Error=',mean_absolute_error(Y, y_pred2))
def rmse(y_pred2, y_roc):
	return np.sqrt(((y_pred2 - y_roc) ** 2).mean())
y_roc = np.array(Y)
rms=rmse(y_pred2,y_roc)
print('\n Root Mean Square Error=',rms)
import pylab as pl
from sklearn.metrics import roc_curve, auc
y_roc = np.array(Y)
fpr, tpr, thresholds = roc_curve(y_roc, y_pred2)
roc_auc = auc(fpr, tpr)
print("\n Area under the ROC curve : %f" % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.show()




t_mat=[]
t_im_unpre = []
t_imm_dwt = []
t_imm_gauss2 = []

for a in range(1,5):
    path=input("Enter the name of the image of current directory"    )
    print(path)
    test = cv2.imread(path)
    cv2.imshow("testimg",test)
    t_resized = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
    t_img_gray = cv2.cvtColor(t_resized, cv2.COLOR_BGR2GRAY)
    t_equ = cv2.equalizeHist(t_img_gray)
    t_equ = t_equ.reshape((576,768))
    t_coeffs = pywt.dwt2(t_equ, 'haar')
    t_equ2 = pywt.idwt2(t_coeffs, 'haar')
    
    bank_gf = createMatchedFilterBank()

    t_equ2 = t_equ2.reshape((576,768))
    t_equ3 = applyFilters(t_equ2,bank_gf)


    imm_test = []

    t_img = t_equ3.reshape((576,768))
    t_Z = t_img.reshape((-1,3))

    # convert to np.float32
    t_Z = np.float32(t_Z)

    t_k=cv2.KMEANS_PP_CENTERS


    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    t_K = 2
    t_ret,t_label,t_center=cv2.kmeans(t_Z,t_K,None,criteria,10,t_k)

    # Now convert back into uint8, and make original image

    t_center = np.uint8(t_center)
    t_res = t_center[t_label.flatten()]
    t_res2 = t_res.reshape((t_img.shape))
    t_res3=cv2.subtract(255,t_res2)
    imm_test.append(np.array(t_res3).flatten())


    #dr2=neigh.predict(imm_test)
    dr = clf.predict(imm_test)
    print(dr)


