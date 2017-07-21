import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime as dt
import math


#this is a dataset of numbers between zero and nine that is squashed down into a vector of
#size 1x784
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#this should create a symbolic vector input for the data set

#None means the dimension can be any length
#W and b are tensors full of zeros
#adding layers
x = tf.placeholder(tf.float32, [None, 784], name="X")
Output_labels = tf.placeholder(tf.float32, [None, 10], name="Output_labels")
def norm_dist_elu(n_inputs, n_outputs):
  return (4/(n_inputs+n_outputs))**(.5)
for training_rate in [1.6e-3,8e-4,4e-4,2e-4,1e-4, 8e-5, 2e-5]:
  for K in [500,300,200,100]:
    for L in [50, 100, 200, 300, 400]:
      if L >= K :
        break
      for M in [30, 50 , 100, 200, 400]:
        if M>=L:
          break
        
        with tf.name_scope("Weights"):
            W1= tf.Variable(tf.truncated_normal([784, K], stddev=norm_dist_elu(784,K)), name="W1")
            tf.summary.histogram("Weights_1", W1)
            
            W2= tf.Variable(tf.truncated_normal([K, L],stddev=norm_dist_elu(K, L)), name = "W2")
            tf.summary.histogram("Weights_2", W2)

            W3= tf.Variable(tf.truncated_normal([L, M],stddev=norm_dist_elu(L,M)),name = "W3")
            tf.summary.histogram("Weights_3", W3)

        ##    W4= tf.Variable(tf.truncated_normal([M, N],stddev=norm_dist_elu(M,N)),name = "W4")
        ##    tf.summary.histogram("Weights_4", W4)

            W_out= tf.Variable(tf.truncated_normal([M, 10],stddev=norm_dist_elu(M, 10)),name = "W_out")
            tf.summary.histogram("Weights_Out", W_out)

        with tf.name_scope("Biases"):
            b1 = tf.Variable(tf.zeros([K]), name= "b1")
            tf.summary.histogram("Biases_1", b1)

            b2 = tf.Variable(tf.zeros([L]),name = "b2")
            tf.summary.histogram("Biases_2", b2)

            b3 = tf.Variable(tf.zeros([M]),name = "b3")
            tf.summary.histogram("Biases_3", b1)

        ##    b4 = tf.Variable(tf.zeros([N]),name = "b4")
        ##    tf.summary.histogram("Biases_4", b4)
            
            b_out= tf.Variable(tf.zeros([10]),name = "b_out")
            tf.summary.histogram("Biases_Out", b_out)



        with tf.name_scope("MultiLayer_NN"):
         #implementing multilayer nn
         y1 = tf.nn.elu(tf.matmul(x, W1) + b1 , name = "y1")
         y2 = tf.nn.elu(tf.matmul(y1, W2) + b2,name =  "y2")
         y3 = tf.nn.elu(tf.matmul(y2, W3) + b3,name = "y3")
        ## y4 = tf.nn.elu(tf.matmul(y3, W4) + b4,name = "y4")
         Output= tf.nn.softmax(tf.matmul(y3, W_out) + b_out, name = "Output")

        #begin defining the cost

        
        with tf.name_scope("Cost"):
          cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Output_labels, logits=Output), name= "Cross_Entropy")

        train_step = tf.train.AdamOptimizer(training_rate).minimize(cross_entropy)


        with tf.Session() as sess:
            init= tf.global_variables_initializer()
            sess.run(init)
            epochs = 2000
            
            now = dt.datetime.utcnow().strftime("%B.%d.%y@%H.%M.%S.%f")
            filestring="/tmp/MNIST_MultiLayer_ANN_Parameter_Optimization/{0},tr={1},fln={2},sln={3},tln={4}".format(now, training_rate, K, L,M)
            print ("Now running ",filestring)
            filewrite_out=tf.summary.FileWriter(filestring)
            filewrite_out.add_graph(sess.graph)
                      
            tf.summary.scalar("Cost", cross_entropy)
            tf.summary.histogram("Cost", cross_entropy)
            
            correct_prediction = tf.equal(tf.argmax(Output, 1), tf.argmax(Output_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar("Accuracy", accuracy)
            tf.summary.histogram("Accuracy", accuracy)

            merged_summaries=tf.summary.merge_all()

            with tf.name_scope("Training"):
                for i in range(epochs):
                #by using small batches of a 100 data points as below, this utilizes stochastic gradient descent
                  batch_xs, batch_ys = mnist.train.next_batch(100)
                  sess.run(train_step, feed_dict={x: batch_xs, Output_labels: batch_ys})

                  
                  if i%5 ==0:
                      sum_op=sess.run(merged_summaries, feed_dict={x: mnist.test.images, Output_labels:mnist.test.labels })
                      filewrite_out.add_summary(sum_op, i)
                  
                  if i%100 ==0:
                    correct_prediction = tf.equal(tf.argmax(Output, 1), tf.argmax(Output_labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print ("The accuracy for run ", i, " in ", epochs, " is ", sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                    Output_labels: mnist.test.labels}))
                    
                
                # Test trained model on dataset
                correct_prediction = tf.equal(tf.argmax(Output, 1), tf.argmax(Output_labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('Accuracy of trained model: ',sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                    Output_labels: mnist.test.labels}))







