import tensorflow as tf
import datetime as datetime


a = tf.placeholder(tf.float32, name ="a")
b = tf.placeholder(tf.float32, name = "b")

now = datetime.datetime.utcnow().strftime("%B.%d.%y@%H.%M.%S.%f")

with tf.name_scope("adder_node"):
    adder_node = a + b #  does exactly what tf.add() does

with tf.Session() as sess:
    filewrite_out=tf.summary.FileWriter("/tmp/Adder_node_example/{}".format(now))
    filewrite_out.add_graph(sess.graph)
