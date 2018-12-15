'''
Since our testing set contains no bounding box labels
we split training set for training and validation,
and use validation set for testing
'''
import numpy as np
import os
from eval_util import evaluate_test_set, get_iou, get_acc, draw_result
from train_fc_regression import final_fc


if __name__ == '__main__':
    TEST_SIZE = 550
#     TEST_SIZE = 3000
    model = final_fc()
    model.load_weights(os.path.join('saved_model', 'fc_regression'))
    
    
    for difficulty in ['easy', 'hard']:
        TEST_FEATURE_FILE = os.path.join('bin_files', 'feature_array_val_{}.bin'.format(difficulty))
        TEST_BBX_FILE = os.path.join('bin_files', 'bbx_array_val_{}.bin'.format(difficulty))
    
        xtest = np.fromfile(TEST_FEATURE_FILE, dtype=np.float32).reshape(TEST_SIZE, -1)
        ltest = np.fromfile(TEST_BBX_FILE, dtype=np.int64).reshape(TEST_SIZE, -1)

        ytest = model.predict(xtest)

        evaluate_test_set(0, TEST_SIZE, ytest, ltest, 'resized_img_val_{}'.format(difficulty))
        print('Percentage of IOU > 0.5 on {0} set:  {1}'.format(difficulty, get_acc(ytest, ltest)))
    
      
#     for difficulty in ['easy']:
#         TEST_FEATURE_FILE = os.path.join('bin_files', 'feature_array_test_{}.bin'.format(difficulty))
#         TEST_BBX_FILE = os.path.join('bin_files', 'bbx_array_test_{}.bin'.format(difficulty))
    
#         xtest = np.fromfile(TEST_FEATURE_FILE, dtype=np.float32).reshape(TEST_SIZE, -1)
#         ltest = np.fromfile(TEST_BBX_FILE, dtype=np.int64).reshape(TEST_SIZE, -1)

#         ytest = model.predict(xtest)

#         evaluate_test_set(0, TEST_SIZE, ytest, ltest, 'resized_img_test_{}'.format(difficulty))
#         print('Percentage of IOU > 0.5 on {0} set:  {1}'.format(difficulty, get_acc(ytest, ltest)))