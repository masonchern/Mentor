import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import heapq

layer_number = int(input('Layer Number: '))
if(layer_number == 1):
  n1 = int(input('Neuron number: '))
if(layer_number == 2):
  n1 = int(input('Neuron number l1 : '))
  n2 = int(input('Neuron number l2 : '))


#case = int(input('which mode(0:average,1:one hot, 2:untouch) : '))
case = 0
regconst=float(input("regularization: "))
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

def compute_accuracy2(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
  acc = 0
  hit = 0
  for i in range(len(y_pre)):
    sec_v = 0
    largest = heapq.nlargest(2,y_pre[i])
    go = 2
    tops = []
    lar = 0
    while(go>0):
      top = [j for j,k in enumerate(y_pre[i]) if k == [largest[lar]]]
      tops = tops+top
      lar = lar + len(tops)
      go = go - len(tops)
      if(len(tops)>2):
        tops = tops[:2]     
    largest = heapq.nlargest(2,v_ys[i])
    label1 = [j for j,k in enumerate(v_ys[i]) if k == [largest[0]]][0]
    label2 = [j for j,k in enumerate(v_ys[i]) if k == [largest[1]]][0]
    if(label1 in tops and label2 in tops):
      hit+=1
    result = hit/len(v_ys)
  return result
#compute accuracy of top 5 one hot model
def compute_accuracy5(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
  acc = 0
  hit = 0
  for i in range(len(y_pre)):
    sec_v = 0
    largest = heapq.nlargest(5,y_pre[i])
    go = 5
    tops = []
    lar = 0
    while(go>0):
      top = [j for j,k in enumerate(y_pre[i]) if k == [largest[lar]]]
      tops = tops+top
      lar = lar + len(tops)
      go = go - len(tops)
      if(len(tops)>5):
        tops = tops[:5]
    largest = heapq.nlargest(2,v_ys[i])
    label1 = [j for j,k in enumerate(v_ys[i]) if k == [largest[0]]][0]
    label2 = [j for j,k in enumerate(v_ys[i]) if k == [largest[1]]][0]
    if(label1 in tops and label2 in tops):
      hit+=1
    result = hit/len(v_ys)
  return result

x_data_train = np.zeros((286*2,1431))
y_data_train = np.zeros((286*2,286))
x_data_test = np.zeros((1430,1431))
y_data_test = np.zeros((1430,286))
#one hot label

for i in range(143):
  y_data_train[i][i] = 1
counter = 0  

for i in range(143,286):
  y_data_train[i][counter] = 1
  counter += 1

counter = 143
for i in range(286,int(286*1.5)):
  y_data_train[i][counter] = 1
  counter += 1

counter = 143
for i in range(int(286*1.5),int(286*2)):
  y_data_train[i][counter] = 1
  counter += 1


counter = -1
counter1 = 143
for i in range(1430):
  if(i%143==0):
    counter += 1
  y_data_test[i][counter] = 1
  y_data_test[i][counter1] = 1
  counter1 += 1
  if(counter1 > 285):
    counter1 = 143


train_data1 = pd.read_csv('../inputs/x_data_train_'+str(chain_num),header=None)
x_data_train1 = train_data1.values

train_data2 = pd.read_csv('../inputs/x_data_train_2',header=None)
x_data_train2 = train_data2.values

x_data_train = np.concatenate((x_data_train1,x_data_train2))


test_data = pd.read_csv('../inputs_mult/x_data_train_mult_1_2',header=None)
x_data_test = test_data.values

if(case == 0):
  x_data_train = x_data_train/196
  x_data_test = x_data_test/196
elif(case == 1):
  x_data_train[x_data_train.nonzero()]= 1 
  x_data_test[x_data_test.nonzero()]= 1 

tempx = x_data_train
tempy = y_data_train

x_data_train = np.concatenate((x_data_test,x_data_train))
y_data_train = np.concatenate((y_data_test,y_data_train))

data_test = pd.read_csv('../inputs/x_data_test_'+str(chain_num),header=None)

x_data_test = data_test/196
y_data_test = tempy[:286]

#xs = tf.placeholder(tf.float32,[None,280280],name='x_input')
xs = tf.placeholder(tf.float32,[None,1431],name='x_input')
ys = tf.placeholder(tf.float32,[None,286],name='y_input')

keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


if(layer_number == 2):
  l1 = add_layer(xs,1431,n1,tf.nn.sigmoid,1)
  l2 = add_layer(l1,n1,n2,tf.nn.sigmoid,1)
  prediction = add_layer(l2,n2,286,tf.nn.softmax,0)
elif(layer_number == 1):
  l1 = add_layer(xs,1431,n1,tf.nn.sigmoid,1)
  prediction = add_layer(l1,n1,286,tf.nn.softmax,0)
elif(layer_number == 0):  
  prediction = add_layer(xs,1431,286,tf.nn.softmax,0)


reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = sum(reg_loss)*regconst
#reg_loss = 0

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
    accuracy_train = compute_accuracy2(x_data_train,y_data_train)

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

