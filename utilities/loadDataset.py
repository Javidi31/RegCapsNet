import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loadDataset(dsname):
    
    if dsname == 'MCYT75':
        img_size = 64
        class_no = 75
        img_per_class = 15
        img_per_class_train = 11
        img_per_class_test = 4
        total_img = 1125
        if True:
            Train = np.zeros(shape=(class_no * img_per_class_train, img_size, img_size))
            Test = np.zeros(shape=(class_no * img_per_class_test, img_size, img_size))
            Train_label = np.zeros(shape=(class_no * img_per_class_train))
            Test_label = np.zeros(shape=(class_no * img_per_class_test))
            Tr_cnt = 0
            Ts_cnt = 0
            for i in range(total_img):
                path = './dataset/MCYT/MCYT75/'
                path = path + str(i // img_per_class + 1)
                path += 'v'
                path += str(i % img_per_class) + '.bmp'

                I = plt.imread(path)
                I = np.reshape(I, (1, img_size, img_size))
                if (i % img_per_class) < img_per_class_train:
                    Train[Tr_cnt, :, :] = I
                    Train_label[Tr_cnt] = i // img_per_class
                    Tr_cnt = Tr_cnt + 1
                else:
                    Test[Ts_cnt, :, :] = I
                    Test_label[Ts_cnt] = i // img_per_class
                    Ts_cnt = Ts_cnt + 1

                if i % 100 == 1:
                    print('image ' + str(i) + " from "+str(total_img))

            Train = (255.0 - Train) / 255.0
            Test = (255.0 - Test) / 255.0

            # print(Train.max())
            # print(Train.min())
            # print(Train.mean())

            Train[Train > 0.15] = 1
            Train[Train <= 0.15] = 0
            Test[Test > 0.15] = 1
            Test[Test <= 0.15] = 0
            with open('./dataset/MCYT/DS_11_4', 'wb') as f:  # 11 train and 4 test
                pickle.dump([Train, Train_label, Test, Test_label], f)

        else:
            with open('./dataset/MCYT/DS_11_4', 'rb') as f:
                Train, Train_label, Test, Test_label = pickle.load(f)
            print('Dataset loaded from file')

    if dsname == 'cedar':
        image_size = 64
        if True:
            from PIL import Image
            Train = np.zeros(shape=(880, image_size, image_size))
            for i in range(880):
                path = './dataset/cedar/cedar_train/'
                path = path + str(i + 1) + '.png'
                # I = plt.imread(path)
                I = Image.open(path)
                I.thumbnail((image_size, image_size), Image.ANTIALIAS)
                I = np.reshape(I, (1, image_size, image_size))
                Train[i, :, :] = I
                if i % 100 == 1:
                    print('train image' + str(i) + "from 880")

            Test = np.zeros(shape=(440, image_size, image_size))
            for i in range(440):
                path = './dataset/cedar/cedar_test/'
                path = path + str(i + 1) + '.png'
                I = plt.imread(path)
                I = Image.open(path)
                I.thumbnail((image_size, image_size), Image.ANTIALIAS)
                I = np.reshape(I, (1, image_size, image_size))
                Test[i, :, :] = I
                if i % 100 == 1:
                    print('test image' + str(i) + "from 440")

            Train = (255.0 - Train) / 255.0
            Test = (255.0 - Test) / 255.0

            print(Train.max())
            print(Train.min())
            print(Train.mean())

            Train[Train > 0.2] = 1
            Train[Train <= 0.2] = 0
            Test[Test > 0.2] = 1
            Test[Test <= 0.2] = 0

            with open('./dataset/cedar/DS_880_440', 'wb') as f:
                pickle.dump([Train, Test], f)
        else:
            with open('./dataset/cedar/DS_880_440', 'rb') as f:
                Train, Test = pickle.load(f)

        Train_label = pd.read_csv('./dataset/cedar/cedarTrain_label.csv', ',',
                                  header=None).values
        Train_label = Train_label[:, 0] - 1

        Test_label = pd.read_csv('./dataset/cedar/cedarTest_label.csv', ',',
                                 header=None).values
        Test_label = Test_label[:, 0] - 1

    