import tensorflow as tf

Weight_Matrix = tf.constant ([ 1,2,3,4,5,6,7, 2,3,4,5,6,7,8, 3,4,5,6,7,8,9 ],shape=[3,7]) 

Vertical_Vector=tf.constant ([ 1,2,3,4,5,6,7],shape=[7,1])

Bias_Vector =tf.constant ([ 1,2,3], shape =[3,1])



output= tf.add(tf.matmul(Weight_Matrix,Vertical_Vector), Bias_Vector)


sess=tf.Session() 
print (sess.run(output))
sess.close()


with tf.Session() as sess:
    print(sess.run(output))
