import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

layer_number = int(input('Layer Number: '))
if(layer_number == 1):
  n1 = int(input('Neuron number: '))
if(layer_number == 2):
  n1 = int(input('Neuron number l1 : '))
  n2 = int(input('Neuron number l2 : '))


#case = int(input('which mode(0:average,1:one hot, 2:untouch) : '))
case = 0

chain_num = int(input('Chain Number : '))

np.set_printoptions(threshold = np.nan)


def add_layer(inputs, in_size, out_size, activation_function=None,drop = 0):
  Weights = tf.Variable(tf.truncated_normal([in_size,out_size]),name='W')
  biases = tf.Variable(tf.zeros([1, out_size])+0.001,name='b')
  Wx_plus_b = tf.add(tf.matmul(inputs, Weights),biases)
  if activation_function is None:
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  if(drop == 1):
    outputs_d = tf.nn.dropout(outputs,keep_prob)
  else:
    outputs_d = outputs
  return outputs_d

#compute accuracy of one hot model
def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
#  print(y_pre)
  correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
  return result

#compute accuracy of regression model

x_data_train = np.zeros((286,1431))
y_data_train = np.zeros((286,143))
x_data_test = np.zeros((286,1431))
y_data_test = np.zeros((286,143))
#one hot label
y_data_train[:143]=np.identity(143)
y_data_train[143:]=np.identity(143)
y_data_test[:143]=np.identity(143)
y_data_test[143:]=np.identity(143)


x_data_train_tran = np.zeros((286,1431))

train_data = pd.read_csv('../inputs/x_data_train_'+str(chain_num),header=None)
x_data_train = train_data.values
test_data = pd.read_csv('../inputs/x_data_test_'+str(chain_num),header=None)
x_data_test = test_data.values


train_data_tran = pd.read_csv('../inputs_tran/x_data_train_'+str(chain_num),header=None)
x_data_train_tran = train_data_tran.values


x_data_train = np.array(x_data_train)
x_data_train_tran = np.array(x_data_train_tran)
if(case == 0):

  x_data_train = (x_data_train+x_data_train_tran)/(196+444)
  x_data_test = x_data_test/196
elif(case == 1):
  x_data_train[x_data_train.nonzero()]= 1 
  x_data_test[x_data_test.nonzero()]= 1 



#xs = tf.placeholder(tf.float32,[None,280280],name='x_input')
xs = tf.placeholder(tf.float32,[None,1431],name='x_input')
ys = tf.placeholder(tf.float32,[None,143],name='y_input')

keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


if(layer_number == 2):
  l1 = add_layer(xs,1431,n1,tf.nn.sigmoid,1)
  l2 = add_layer(l1,n1,n2,tf.nn.sigmoid,1)
  prediction = add_layer(l2,n2,143,tf.nn.softmax,0)
elif(layer_number == 1):
  l1 = add_layer(xs,1431,n1,tf.nn.sigmoid,1)
  prediction = add_layer(l1,n1,143,tf.nn.softmax,0)
elif(layer_number == 0):  
  prediction = add_layer(xs,1431,143,tf.nn.softmax,0)


reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = sum(reg_loss)*0.00001

#fixing some possible numeric error version
cross_entropy = reg_loss + tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction,1e-30,1.0)),reduction_indices=[1]))

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

#loss = tf.reduce_mean(tf.squared_difference(prediction,ys))


train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

#final learning rate
min_learning_rate = 0.000001
#initial learning rate
max_learning_rate = 0.001

decay_speed = 25000.0 
#drop out probability
drop_rate = 0.93

saver = tf.train.Saver()

best_ac = 0
counter = 0

for i in range(25000):
  learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)
#  learning_rate = 0.0001
  sess.run(train_step,feed_dict = {xs: x_data_train, ys: y_data_train, keep_prob: drop_rate, lr: learning_rate})

  if(i%20==0):
    accuracy_train = compute_accuracy(x_data_train,y_data_train)
    accuracy_test = compute_accuracy(x_data_test,y_data_test)

    if(accuracy_train>0.999):
      print('Accuracy to 1')
      save_path = saver.save(sess,'model.ckpt')
      break

    if(accuracy_train>best_ac):
      counter = 0   
      best_ac = accuracy_train 
      save_path = saver.save(sess,'model.ckpt')
      print('Accuracy to', best_ac)
    else:
      counter += 1
      if(counter>20):
        break   

saver.restore(sess,"model.ckpt")


print('compute final')
print(compute_accuracy(x_data_test,y_data_test))

