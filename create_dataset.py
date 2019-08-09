import os
import shutil
from scipy import io

batch_size = 64
data_path = '/media/Data/Datasets/Dogs_Breed/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'

if not os.path.exists(train_dir) :
    os.makedirs(train_dir)

if not os.path.exists(test_dir) :
    os.makedirs(test_dir)

test_filenames = io.loadmat(data_path + 'lists/test_list.mat')['annotation_list']
train_filenames = io.loadmat(data_path + 'lists/train_list.mat')['annotation_list']

print('Creation du dataset test')
for i in range(test_filenames.shape[0]) :
    print('\r{}/{}'.format(i+1, test_filenames.shape[0]), end='')
    filename = test_filenames[i][0][0]
    dirname = filename.split('/')[0]

    if not os.path.exists(test_dir + dirname):
        os.makedirs(test_dir + dirname)

    if not os.path.exists(train_dir + filename + '.jpg') :
        shutil.copyfile(data_path + 'Images/' + filename + '.jpg', test_dir + filename + '.jpg')
print()

print('Creation du dataset train')
for i in range(train_filenames.shape[0]) :
    print('\r{}/{}'.format(i+1, train_filenames.shape[0]), end='')
    filename = train_filenames[i][0][0]
    dirname = filename.split('/')[0]

    if not os.path.exists(train_dir + dirname):
        os.makedirs(train_dir + dirname)
    if not os.path.exists(train_dir + filename + '.jpg') :
        shutil.copyfile(data_path + 'Images/' + filename + '.jpg', train_dir + filename + '.jpg')
print()
