
# coding: utf-8

# # Continuation of the previous model. 
# ## Major changes in this model:
# <ol> 
#     <li> Performing the change of origin to make all the data values positive (change will be done in preprocessor script) </li>
#     
#     <li> Apply the Absolute Neural Network (bidirectional model) for this problem </li>
# </ol>

# # Model 2

# As usual, I start with the utility cells

# In[1]:


# packages used for processing:
import cPickle as pickle # for pickling the processed data
import matplotlib.pyplot as plt # for visualization
import numpy as np # numerical computations

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# the boss of tensorflow frameworks
import tensorflow as tf

# to plot the images inline
get_ipython().magic(u'matplotlib inline')


# In[2]:


# Input data files are available in the "../Data/" directory.

def exec_command(cmd):
    '''
        function to execute a shell command and see it's 
        output in the python console
        @params
        cmd = the command to be executed along with the arguments
              ex: ['ls', '../input']
    '''
    print(check_output(cmd).decode("utf8"))


# In[3]:


# check the structure of the project directory
exec_command(['ls', '..'])


# In[4]:


''' Set the constants for the script '''

# various paths of the files
data_path = "../Data" # the data path
base_model_path = "../Models"

data_files = {
    "train": os.path.join(data_path, "train.csv"),
    "test": os.path.join(data_path, "test.csv")
}

base_model_path = '../Models'

plug_and_play_data_file_path = os.path.join(data_path, "plug_and_play_for_ANN.pickle")

# constants:
(train_size, dev_size, test_size) = (0.9, 0.05, 0.05) # values are unit ratios
no_of_features = 57
no_of_itreations = 10000 
batch_size = 512
checkpoint_factor = 50
lr = 1e-6


# In[5]:


# function to unpickle the given file and load the obj back into the python environment
def unPickleIt(pickle_path): # might throw the file not found exception
    '''
        function to unpickle the object from the given path
        @param
        pickle_path => the path where the pickle file is located
        @return => the object extracted from the saved path
    '''

    with open(pickle_path, 'rb') as dumped_pickle:
        obj = pickle.load(dumped_pickle)

    return obj # return the unpickled object


# # Load in the data and create the train / dev / test splits

# In[6]:


data_dict = unPickleIt(plug_and_play_data_file_path)


# In[7]:


X = data_dict['features']; Y = data_dict['labels']


# In[8]:


X.shape, Y.shape # check if the shapes are compatible


# In[9]:


# keep the variances for the features
variances = data_dict['variances']


# In[10]:


# function to split the data into train, dev and test sets
def train_dev_test_split_data(X, Y):
    '''
        function to split the X and Y arrays into train, dev and test sets
        @param
        X => the input features to train on
        Y => the ideal labels for the given inputs
        @return => train_X, train_Y, dev_X, dev_Y, test_X, test_Y: the names suggest meanings
    '''
    m_examples = X.shape[-1] # total number of examples to train on
    
    # first parition point
    train_dev_partition_point = int((m_examples * train_size) + 0.5)
    
    # second partition point 
    dev_test_partition_point = train_dev_partition_point + int((m_examples * dev_size) + 0.5)
    
    ''' perform the actual split of the data '''
    # Training set splitting:
    train_X = X[:, : train_dev_partition_point]; train_Y = Y[:, : train_dev_partition_point]
    
    # dev set splitting
    dev_X = X[:, train_dev_partition_point: dev_test_partition_point]
    dev_Y = Y[:, train_dev_partition_point: dev_test_partition_point]
    
    # test set splitting
    test_X = X[:, dev_test_partition_point:]; test_Y = Y[:, dev_test_partition_point:]
    
    # return the so formed splits
    return train_X, train_Y, dev_X, dev_Y, test_X, test_Y


# In[11]:


train_X, train_Y, dev_X, dev_Y, test_X, test_Y = train_dev_test_split_data(X, Y)


# In[12]:


# print the shapes of all the above obtained sets:
print "Training X shape: " + str(train_X.shape)
print "Training Y shape: " + str(train_Y.shape)
print "Dev X shape     : " + str(dev_X.shape)
print "Dev Y shape     : " + str(dev_Y.shape)
print "Test X shape    : " + str(test_X.shape)
print "Test Y shape    : " + str(test_Y.shape)


# In[13]:


# Make sure that no Example has been left out
assert X.shape[-1] == np.hstack((train_X, dev_X, test_X)).shape[-1], "Examples have been left out"
assert Y.shape[-1] == np.hstack((train_Y, dev_Y, test_Y)).shape[-1], "Labels have been left out"

# If both the above asserts are successful, we can go ahead and print the following statement
print "Both the assertions pass!!"


# # Cool! So now Let's get onto the part where we build the Tensorflow Graph
# -------------------------------------------------------------------------------------------------------------------
# ## I am going to keep the graph scoped and in a single cell, so that I can port it into the production graph file

# In[14]:


# the num_units in each layer of the feed_forward neural network
layer_dims = [512, 512, 512, 512, 512, 512, 512, 512, 2]


# In[15]:


train_Y.shape


# ### Use this point to restart the graph building process!

# In[16]:


tf.reset_default_graph()


# In[17]:


# scoped as Inputs
with tf.variable_scope("Input"):
   # define the placeholders for the input data
   # placeholder for feeding in input data batch
   input_X = tf.placeholder(tf.float32, shape=(None, no_of_features), name="Input_features")
   labels_Y = tf.placeholder(tf.int32, shape=(None,), name="Ideal_labels") # placeholder for the labels
   one_hot_encoded_labels_Y = tf.one_hot(labels_Y, depth=2, axis=1, name="One_hot_label_encoder")


# In[18]:


one_hot_encoded_labels_Y


# In[19]:


# scoped as model:
with tf.variable_scope("Deep_Neural_Network"):
    # define the layers for the neural network.
    with tf.name_scope("Encoder"):
        ''' This is The forward-backward neural network with abs activation function '''
        # layer 1 => 
        fwd_lay1 = tf.layers.dense(input_X, layer_dims[0], activation=tf.abs, name="layer_1")
        # layer 2 =>
        fwd_lay2 = tf.layers.dense(fwd_lay1, layer_dims[1], activation=tf.abs, name="layer_2")
        # layer 3 =>
        fwd_lay3 = tf.layers.dense(fwd_lay2, layer_dims[2], activation=tf.abs, name="layer_3")
        # layer 4 =>
        fwd_lay4 = tf.layers.dense(fwd_lay3, layer_dims[3], activation=tf.abs, name="layer_4")
        # layer 5 =>
        fwd_lay5 = tf.layers.dense(fwd_lay4, layer_dims[4], activation=tf.abs, name="layer_5")
        # layer 6 =>
        fwd_lay6 = tf.layers.dense(fwd_lay5, layer_dims[5], activation=tf.abs, name="layer_6")
        # layer 7 =>
        fwd_lay7 = tf.layers.dense(fwd_lay6, layer_dims[6], activation=tf.abs, name="layer_7")
        # layer 8 =>
        fwd_lay8 = tf.layers.dense(fwd_lay7, layer_dims[7], activation=tf.abs, name="layer_8")
        # layer 9 =>
        fwd_lay9 = tf.layers.dense(fwd_lay8, layer_dims[8], activation=tf.abs, name="layer_9")
        
    ''' Separately record all the activations as histograms '''
    # recording the summaries to visualize separately
    fwd_lay1_summary = tf.summary.histogram("fwd_lay1_summary", fwd_lay1)
    fwd_lay2_summary = tf.summary.histogram("fwd_lay2_summary", fwd_lay2)
    fwd_lay3_summary = tf.summary.histogram("fwd_lay3_summary", fwd_lay3)
    fwd_lay4_summary = tf.summary.histogram("fwd_lay4_summary", fwd_lay4)
    fwd_lay5_summary = tf.summary.histogram("fwd_lay5_summary", fwd_lay5)
    fwd_lay6_summary = tf.summary.histogram("fwd_lay6_summary", fwd_lay6)
    fwd_lay7_summary = tf.summary.histogram("fwd_lay7_summary", fwd_lay7)
    fwd_lay8_summary = tf.summary.histogram("fwd_lay8_summary", fwd_lay8)
    fwd_lay9_summary = tf.summary.histogram("fwd_lay9_summary", fwd_lay9)


# In[20]:


with tf.variable_scope("", reuse=True):
    # bring out all the weights from the network
    lay_1_wts = tf.get_variable("Deep_Neural_Network/layer_1/kernel")
    lay_2_wts = tf.get_variable("Deep_Neural_Network/layer_2/kernel")
    lay_3_wts = tf.get_variable("Deep_Neural_Network/layer_3/kernel")
    lay_4_wts = tf.get_variable("Deep_Neural_Network/layer_4/kernel")
    lay_5_wts = tf.get_variable("Deep_Neural_Network/layer_5/kernel")
    lay_6_wts = tf.get_variable("Deep_Neural_Network/layer_6/kernel")
    lay_7_wts = tf.get_variable("Deep_Neural_Network/layer_7/kernel")
    lay_8_wts = tf.get_variable("Deep_Neural_Network/layer_8/kernel")
    lay_9_wts = tf.get_variable("Deep_Neural_Network/layer_9/kernel")
    
    lay_1_biases = tf.get_variable("Deep_Neural_Network/layer_1/bias")
    lay_2_biases = tf.get_variable("Deep_Neural_Network/layer_2/bias")
    lay_3_biases = tf.get_variable("Deep_Neural_Network/layer_3/bias")
    lay_4_biases = tf.get_variable("Deep_Neural_Network/layer_4/bias")
    lay_5_biases = tf.get_variable("Deep_Neural_Network/layer_5/bias")
    lay_6_biases = tf.get_variable("Deep_Neural_Network/layer_6/bias")
    lay_7_biases = tf.get_variable("Deep_Neural_Network/layer_7/bias")
    lay_8_biases = tf.get_variable("Deep_Neural_Network/layer_8/bias")
    lay_9_biases = tf.get_variable("Deep_Neural_Network/layer_9/bias")


# In[21]:


lay_1_wts, lay_8_wts, lay_9_wts, lay_1_biases, lay_8_biases, lay_9_biases


# In[22]:


y_back_in = fwd_lay9
y_back_in


# In[23]:


with tf.name_scope("Decoder"):
        lay_0_biases = tf.get_variable("layer_0/bias", shape=(no_of_features, ))
    
        # layer 1 => 
        bwd_lay1 = tf.abs(tf.matmul(y_back_in, tf.transpose(lay_9_wts)) + lay_8_biases)
        # layer 2 => 
        bwd_lay2 = tf.abs(tf.matmul(bwd_lay1, tf.transpose(lay_8_wts)) + lay_7_biases)
        # layer 3 => 
        bwd_lay3 = tf.abs(tf.matmul(bwd_lay2, tf.transpose(lay_7_wts)) + lay_6_biases)
        # layer 4 => 
        bwd_lay4 = tf.abs(tf.matmul(bwd_lay3, tf.transpose(lay_6_wts)) + lay_5_biases)
        # layer 5 => 
        bwd_lay5 = tf.abs(tf.matmul(bwd_lay4, tf.transpose(lay_5_wts)) + lay_4_biases)
        # layer 6 => 
        bwd_lay6 = tf.abs(tf.matmul(bwd_lay5, tf.transpose(lay_4_wts)) + lay_3_biases)
        # layer 7 => 
        bwd_lay7 = tf.abs(tf.matmul(bwd_lay6, tf.transpose(lay_3_wts)) + lay_2_biases)
        # layer 8 => 
        bwd_lay8 = tf.abs(tf.matmul(bwd_lay7, tf.transpose(lay_2_wts)) + lay_1_biases)
        # layer 9 => 
        bwd_lay9 = tf.abs(tf.matmul(bwd_lay8, tf.transpose(lay_1_wts)) + lay_0_biases)


# In[24]:


x_back_out = bwd_lay9
x_back_out, input_X


# In[25]:


# function to compute the directional cosines of the input values
def directional_cosines(X):
    ''' 
        calculate the directional cosines of the inputs
    '''
    square = tf.square(X)
    sum_square = tf.reduce_sum(square, axis=1, keep_dims=True)
    dcs = X / tf.sqrt(sum_square)
    
    # return the directional cosines:
    return dcs


# In[26]:


# scoped as predictions
with tf.variable_scope("Prediction"):
    prediction = directional_cosines(y_back_in)


# In[27]:


# scoped as loss
with tf.variable_scope("Loss"):
    # define the forward loss
    fwd_loss = tf.reduce_mean(tf.abs(prediction - one_hot_encoded_labels_Y))
    
    # define the reverse loss
    rev_loss = tf.reduce_mean(tf.abs(x_back_out - input_X))
    
    total_loss = fwd_loss + rev_loss
        
    # record the loss summary:
    tf.summary.scalar("Fwd_loss", fwd_loss)
    tf.summary.scalar("Bwd_loss", rev_loss)
    tf.summary.scalar("Tot_loss", total_loss)


# In[28]:


# scoped as train_step
with tf.variable_scope("Train_Step"):
    # define the optimizer and the train_step:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr) # this has been manually tuned
    train_step = optimizer.minimize(total_loss, name="train_step")


# In[29]:


# scoped as init operation
with tf.variable_scope("Init"):
    init_op = tf.global_variables_initializer()


# In[30]:


# scoped as summaries
with tf.variable_scope("Summary"):
    all_summaries = tf.summary.merge_all()


# # The graph has been defined. Now, use the session executer to run the graph and see how it trains.

# In[31]:


model_name = "Model2"


# In[32]:


# function to execute the session and train the model:
def execute_graph(dataX, dataY, exec_graph, model_name, no_of_iterations):
    '''
        function to start and execute the session with training.
        @param 
        dataX, dataY => the data to train on
        exec_graph => the computation graph to be trained
        model_name => the name of the model where the files will be saved
        no_of_itreations => no of iterations for which the model needs to be trained
        @return => Nothing, this function has a side effect
    '''
    assert dataX.shape[-1] == dataY.shape[-1], "The Dimensions of input X and labels Y don't match"
    
    # the number of examples in the dataset
    no_of_examples = dataX.shape[-1]
    
    with tf.Session(graph=exec_graph) as sess:
        # create the tensorboard writer for collecting summaries:
        log_dir = os.path.join(base_model_path, model_name)
        tensorboard_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph, filename_suffix=".bot")
        
        # The saver object for saving and loading the model
        saver = tf.train.Saver(max_to_keep=2)
        
        # check if the model has been saved.
        model_path = log_dir
        model_file = os.path.join(model_path, model_name) # the name of the model is same as dir
        if(os.path.isfile(os.path.join(base_model_path, model_name, "checkpoint"))):
            # the model exists and you can restore the weights
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            # no saved model found. so, run the global variables initializer:
            sess.run(init_op)

        print "Starting the training ..."
        print "==============================================================================================="
        
        batch_index = 0 # initialize it to 0
        # start the training:
        for iteration in range(no_of_itreations):
            
            # fetch the input and create the batch:
            start = batch_index; end = start + batch_size
            inp_X = dataX[:, start: end].T # extract the input features
            inp_Y = dataY[:, start: end].T # extract the labels
            
            # feed the input to the graph and get the output:
            _, cost = sess.run((train_step, total_loss), feed_dict={input_X: inp_X, labels_Y: np.squeeze(inp_Y)})
            
            # checkpoint the model at certain times
            if((iteration + 1) % checkpoint_factor == 0):
                # compute the summary:
                summary = sess.run(all_summaries, feed_dict={input_X: inp_X, labels_Y: np.squeeze(inp_Y)})
                
                # accumulate the summary
                tensorboard_writer.add_summary(summary, (iteration + 1))
                
                # print the cost at this point
                print "Iteration: " + str(iteration + 1) + " Current cost: " + str(cost)
                
                # save the model trained so far:
                saver.save(sess, model_file, global_step = (iteration + 1))
                
            # increment the batch_index
            batch_index = (batch_index + batch_size) % no_of_examples
            
        print "==============================================================================================="
        print "Training complete"


# In[36]:


# use the above defined method to start the training:
execute_graph(train_X, train_Y, tf.get_default_graph(), model_name, 100)


# # Calculate the accuracy on the dev set

# In[50]:


def calc_accuracy(dataX, dataY, exec_graph, model_name, threshold = 0.5):
    '''
        Function to run the trained model and calculate it's accuracy on the given inputs
        @param 
        dataX, dataY => The data to be used for accuracy calculation
        exec_graph => the Computation graph to be used
        model_name => the model to restore the weights from
        threshold => the accuracy threshold (by default it is 0.5)
        @return => None (function has side effect)
    '''
    assert dataX.shape[-1] == dataY.shape[-1], "The Dimensions of input X and labels Y don't match"
    
    # the number of examples in the dataset
    no_of_examples = dataX.shape[-1]
    
    with tf.Session(graph=exec_graph) as sess:
        
        # The saver object for saving and loading the model
        saver = tf.train.Saver(max_to_keep=2)
        
        # the model must exist and you must be able to restore the weights
        model_path = os.path.join(base_model_path, model_name)
        assert os.path.isfile(os.path.join(model_path, "checkpoint")), "Model doesn't exist"
        
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        
        # compute the predictions given out by model
        preds = sess.run(prediction, feed_dict={input_X: dataX.T, labels_Y: np.squeeze(dataY.T)})
        print preds.shape
        print preds[100: 120, :]
        
        label_preds = np.argmax(preds, axis=1)
        
        # calculate the accuracy in percentage:
        correct = np.sum((label_preds == np.squeeze(dataY.T)))
        accuracy = (float(correct) / dataX.shape[-1]) * 100 # for percentage
        
    # return the so calculated accuracy:
    return accuracy


# In[45]:


print "Train_Set Accuracy: " + str(calc_accuracy(train_X, train_Y, tf.get_default_graph(), model_name))


# In[51]:


print "Dev Set Accuracy: " + str(calc_accuracy(dev_X, dev_Y, tf.get_default_graph(), model_name))


# # The model doesn't show any new promise, but the network has indeed been trained as per the forward-backward architecture. Let's see what happens now. (Although the accuracy results are just as they were earlier)
