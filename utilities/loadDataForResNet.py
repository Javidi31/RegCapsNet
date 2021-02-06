"""
Created on Wed Jun 12 11:01:10 2019

@author: Malihe Javidi, m.javidi@qiet.ac.ir
"""

def loadDataForResNet():

path1        = '.\\dataset\\Cedar\\'  #######
files = os.listdir(path1);
numberOfsamples = len(files)
all_tr_images  = []
all_te_images  = []
y_train     = []
y_test     = []

img_rows, img_cols = 200, 200
img_channels    = 1
num_classes     = 55  #######
img_no_perClass = 24  #######  
train_no        = 6   #######
test_no         = 18  #######
lb_no = 0

for number1 in range(0, numberOfsamples,img_no_perClass):    
    print(number1)    
    for number2 in range(0, train_no):        
        path2 = path1 + files[number1]
        img = cv2.imread(path2 , 0)
        img = img.reshape(img_rows, img_cols, 1)
        all_tr_images.append(img)    
        y_train.append(lb_no)
        number1 = number1+1
    for number2 in range(0, test_no):
        path2 = path1 + files[number1]
        img = cv2.imread(path2 , 0)
        img = img.reshape(img_rows, img_cols, 1)
        all_te_images.append(img)    
        y_test.append(lb_no)
        number1 = number1+1
    lb_no = lb_no + 1

X_train = np.array(all_tr_images)
y_train = np.array(y_train, dtype=np.uint16) 
print(X_train.shape)

X_test = np.array(all_te_images)
y_test = np.array(y_test, dtype=np.uint16) 
print(X_test.shape)

# Convert class vectors to multi class matrices.
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test  = np_utils.to_categorical(y_test, num_classes)

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train   -= mean_image
X_test    -= mean_image
X_train   /= 128.
X_test    /= 128.