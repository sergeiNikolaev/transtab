import os
os.chdir('cell_properties_learning')

import transtab
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


### global parameters
GENERAL_EXAMPLES_AMOUNT = 20


# set random seed
transtab.random_seed(42)

# load a dataset and start vanilla supervised training
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data()

### create x_test and y_test from lines where the result is one
x_test_list = []
columns_data = testset[0][0].columns
y_test_dict = {}
for index in range(0, len(testset)):
    for element in testset[index][1].keys():
        if testset[index][1][element] == 1:
            x_test_list.append(testset[index][0].loc[element,:])
            y_test_dict[element] = testset[index][1][element]
        elif GENERAL_EXAMPLES_AMOUNT > 0:
            x_test_list.append(testset[index][0].loc[element, :])
            y_test_dict[element] = testset[index][1][element]
            GENERAL_EXAMPLES_AMOUNT = GENERAL_EXAMPLES_AMOUNT - 1
x_test = pd.DataFrame(x_test_list, columns=columns_data)
y_test = pd.Series(data = y_test_dict, index = y_test_dict.keys())

# build transtab classifier model
model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

# start training
training_arguments = {
    'num_epoch':50,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint',
    'batch_size':128,
    'lr':1e-4,
    'weight_decay':1e-4,
    }
training_loss = transtab.train(model, trainset, valset, **training_arguments)

x_data = []
y_data = []
for time_index in training_loss.keys():
    x_data.append(time_index)
    y_data.append(training_loss[time_index])

markers_list = ['d', 'v', '1', 'X', 'x', 'P', '+', 'o', '.', '*']
col = sns.color_palette("tab10", n_colors=200)

filename = "training_loss.png"
fig = plt.figure(1000, figsize = (10, 10))
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data, color = col[0], marker = markers_list[9], s = 30)  # , label=name)
ax.plot(x_data, y_data, color = col[0], linewidth = 2, alpha = 0.2)
plt.savefig(filename, bbox_inches='tight')
plt.close()
#plt.show()

filename1 = "training_loss_without_first.png"
fig1 = plt.figure(1000, figsize=(10, 10))
ax1 = fig1.add_subplot(111)
x_data_copy = []
y_data_copy = []
for index in range(0, len(x_data)):
    if x_data[index] != 0:
        x_data_copy.append(x_data[index])
        y_data_copy.append(y_data[index])
ax1.scatter(x_data_copy, y_data_copy, color = col[3], marker = markers_list[3], s = 30)  # , label=name)
ax1.plot(x_data_copy, y_data_copy, color = col[3], linewidth = 2, alpha = 0.2)
plt.savefig(filename1, bbox_inches='tight')
plt.close()


# save model
model.save('./ckpt/pretrained')


# evaluation
ypred = transtab.predict(model, x_test, y_test)
prediction_index = 0
for element in y_test.keys():
    print("The result has to be: " + str(y_test[element]) + " and it is: " + str(ypred[prediction_index]))
    prediction_index = prediction_index + 1
print("Unbelivable results!!!")


transtab.evaluate(ypred, y_test, metric='auc')
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, ypred))


# # now let's use another data and try to leverage the pretrained model for finetuning
# # here we have loaded the required data `credit-approval` before, no need to load again.

# # load the pretrained model
# # model.load('./ckpt/pretrained')

# update model's categorical/numerical/binary column dict
# need to specify the number of classes if the new dataset has different # of classes from the
# pretrained one.
# model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols, 'num_class':2})
#
#
# # start training
# training_arguments = {
#     'num_epoch':50,
#     'eval_metric':'auc',
#     'eval_less_is_better':False,
#     'output_dir':'./checkpoint',
#     'batch_size':128,
#     'lr':2e-4,
#     }
#
# transtab.train(model, trainset[1], valset[1], **training_arguments)
#


