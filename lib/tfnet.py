#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pkumar

Class for creating, training and testing a fully connected general deep neural 
network using tensorflow!

Main reason to implement the code is here is to see if this increases the speed
and training as compared to Keras implementation.

Also want to see if we can get COMPLETE reproducibility! Keras is hacky at 
sometimes.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import math

"""
Creating a tensorflow class
"""
class tfnet():
  
  """
  Initializer function
  Pass data and options while creating the network
  Data contains the entire data set as a dictionary
  {X: X, Y: Y}
  options can contain options for the network such as:
  layer_dims: contains number of nodes in each layer
  frac_train: fraction of total data used for training
  learning_rate: learning_rate
  num_epochs: Number of epochs
  optimization_algorithm: type of optimizer, by default, adam
  activation: relu or tanh
  minibatch_size: mini batch size
  print_cost: True or False (used to print the cost function during training)
  np_seed: numpy seed
  tf_seed: seed for tensorflow
  ---------------------
  At the end this init function will create a tf dictionary which will store 
  tensors and temporary variables need DURING implementation. 
  This is easier for passsing variables around functions!
  """
  def __init__(self,data,options):
    self.data=data
    self.options=options
    np.random.seed(self.options['np_seed'])
    tf.set_random_seed(self.options['tf_seed'])
    self.cache={}
    
    
  """
  Create tensorflow placeholders for training and testing dataset.
  """  
  def create_placeholders(self,n_x,n_y):
    
    
    X=tf.placeholder(tf.float64,shape=(n_x,None))
    Y=tf.placeholder(tf.float64,shape=(n_y,None))   
    
    return X,Y
  
  
  """
  Function to initialize the parameter weight matrices
  """
  def initialize_parameters(self):
    

    # Extracting some options
    layer_dims=self.options['layer_dims']
    
    # Extract total number of layers
    L=len(layer_dims)
    
    # Initializing a dict for parameters
    parameters={}
    
    # Checking if an initialization option has been passed
    # If not, do random initialization! 
    try:
      initialization=self.options['initialization']
    except:
      initialization='random'
    
    for l in range(1,L):
      
      # Checking the value of init
      if initialization=='random':
        parameters["W"+str(l)]=tf.get_variable("W"+str(l),
                        [layer_dims[l],layer_dims[l-1]],
                        tf.float64, 
                        initializer=tf.contrib.layers.xavier_initializer(
                                seed=self.options['tf_seed']))
      
        parameters["b"+str(l)]=tf.get_variable("b"+str(l),
                        [layer_dims[l],1],
                        tf.float64, 
                        initializer=tf.zeros_initializer())
      
      # If we have custom weights
      elif initialization=='custom':
	
        parameters["W"+str(l)]=tf.Variable(self.options['initialWeights']["W"+str(l)],
                                           name="W"+str(l),dtype=tf.float64)
      
        parameters["b"+str(l)]=tf.Variable(self.options['initialWeights']["b"+str(l)],
                                           name="b"+str(l),dtype=tf.float64)
        
    return parameters

    
  """
  Forward propagation for the computation graph
  """
  def forward_propagation(self,X,parameters):
      
    # Extracting some options
    layer_dims=self.options['layer_dims']
    act=self.options['activation']
    
    # Extract total number of layers
    L=len(layer_dims)
          
    self.cache["A0"]=X
    for l in range(1,L-1):
        
      W=parameters["W"+str(l)]
      b=parameters["b"+str(l)]
      self.cache["Z"+str(l)]=tf.add(tf.matmul(W,self.cache["A"+str(l-1)]),b)
        
      if act=='tanh':
        self.cache["A"+str(l)]=tf.nn.tanh(self.cache["Z"+str(l)])
      elif act=='abs':
        self.cache["A"+str(l)]=tf.abs(self.cache["Z"+str(l)])                
      else:
        self.cache["A"+str(l)]=tf.nn.relu(self.cache["Z"+str(l)])
    
    # This is the final layer
    W=parameters["W"+str(L-1)]
    b=parameters["b"+str(L-1)]
    ZL=tf.add(tf.matmul(W,self.cache["A"+str(L-2)]),b)

    return ZL
  
  """
  Forward propagation for prediction
  """
  def forward_propagation_for_prediction(self,X,parameters):
    
    # Extracting some options
    layer_dims=self.options['layer_dims']
    act=self.options['activation']
    
    # Extract total number of layers
    L=len(layer_dims)

    A=X
    for l in range(1,L-1):
        
      W=parameters["W"+str(l)]
      b=parameters["b"+str(l)]
      Z=np.dot(W,A)+b
        
      if act=='tanh':
        A=np.tanh(Z)
      elif act=='abs':
        A=np.absolute(Z)              
      else:
        A=Z*(Z>0)
    
    # This is the final layer
    W=parameters["W"+str(L-1)]
    b=parameters["b"+str(L-1)]
    ZL=np.dot(W,A)+b
    
    
    # Converting this into original scale
    #Y_mean=self.cache['Y_mean']
    #Y_std=self.cache['Y_std']
    #ZL=ZL*Y_std+Y_mean
    
    assert (ZL.shape == (W.shape[0],X.shape[1]))
        
    return ZL

    
  
  """
  Computes Cost
  """
  def compute_cost(self,Z,Y,parameters):
    
    try:        
      self.options['regularization']
    except:
      self.options['regularization']=False

    if self.options['regularization']:
      cost = tf.reduce_mean(tf.squared_difference(Z, Y))
      cost = cost + tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2'])
      cost = cost + tf.nn.l2_loss(parameters['b1']) + tf.nn.l2_loss(parameters['b2'])
    else:
      cost = tf.reduce_mean(tf.squared_difference(Z, Y))
    return cost
  
  
  """
  This function is used to split the entire data set into training and 
  test data set. 
  
  This also normalizes the data set!
  """
  def data_preprocessing(self):
    
    frac_split=self.options['frac_train']
    X=self.data['X']
    Y=self.data['Y']
    
    num_train=math.floor(frac_split*X.shape[1])
    
    # This was the best I could think for shuffling the columns!
    idx=np.arange(X.shape[1])
    np.random.shuffle(idx)

    # First of all shuffle the data set!
    X=X[:,idx]
    Y=Y[:,idx]
    
    # Training data set
    X_train=X[:,0:num_train]
    Y_train=Y[:,0:num_train]
    X_test=X[:,num_train:]
    Y_test=Y[:,num_train:]
    
    if self.options['normalize_data']:
      # Compute the mean and variance of the training data set
      X_mean=np.mean(X_train,axis=1).reshape(X.shape[0],1)
      X_std=np.std(X_train,axis=1).reshape(X.shape[0],1)
      Y_mean=np.mean(Y_train,axis=1).reshape(Y.shape[0],1)
      Y_std=np.std(Y_train,axis=1).reshape(Y.shape[0],1)
    
      # Some assert statements
      assert (X_mean.shape==(X.shape[0],1))
      assert (X_std.shape==(X.shape[0],1))
      assert (Y_mean.shape==(Y.shape[0],1))
      assert (Y_std.shape==(Y.shape[0],1))
    
      # Compute the final data used for training and testing     
      X_train=(X_train-X_mean)/X_std
      Y_train=(Y_train-Y_mean)/Y_std
      X_test=(X_test-X_mean)/X_std
      Y_test=(Y_test-Y_mean)/Y_std

      # Storing the mean and variances becuase they are required later
      # These are required even during prediction or will be required during 
      # closed loop simulations!
      self.cache['X_mean']=X_mean
      self.cache['X_std']=X_std
      self.cache['Y_mean']=Y_mean
      self.cache['Y_std']=Y_std
    
    return X_train, Y_train, X_test, Y_test
  
  
  """
  Creates a list of random minibatches.  
  """
  def random_mini_batches(self,X,Y,mini_batch_size=64,seed=0):
    
    m = X.shape[1]                  # number of examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # Number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)    
    
    
    return mini_batches
  
  
  """
  This is the main function to build and train the model!
  """
  def model(self):
    
    # Some preliminary stuff
    ops.reset_default_graph()
    tf.set_random_seed(self.options['tf_seed'])
    seed=self.options['np_seed']
    
    X_train, Y_train, X_test, Y_test=self.data_preprocessing()
    (n_x,m)=X_train.shape
    n_y=Y_train.shape[0]

    # Lists to save the costs and parameters
    costs=[]
    parametersItr=[]
    
    # Extracting some options
    num_epochs=self.options['num_epochs']
    learning_rate=self.options['learning_rate']
    minibatch_size=self.options['minibatch_size']
    print_cost=self.options['print_cost']
    
    # Create Placeholders
    X,Y=self.create_placeholders(n_x,n_y)
    
    # Initialize parameters
    parameters=self.initialize_parameters()
    
    # Forward Propagation, Create the tensorflow graph
    ZL=self.forward_propagation(X,parameters)
    
    # Add cost function to the computation graph
    cost=self.compute_cost(ZL,Y,parameters)

    # Define the tensorflow optimizer
    if self.options['optimization_algorithm']=='adam':
      optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    if self.options['optimization_algorithm']=='sgd':
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all variables
    init=tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
      
      # Run the initialization
      sess.run(init)
      
      # Just seeing the initial parameters 
      print('Initial Cost: ' + str(cost.eval({X:X_train,Y:Y_train})))
      parametersInit=sess.run(parameters)
      print('W1:' + str(parametersInit['W1']))

      # Just checking the initial cost
      print('Initial Cost: ' + str(cost.eval({X:X_train,Y:Y_train})))
      
      # Do the training loop
      for epoch in range(1,num_epochs+1):
        
        seed=seed+1
        minibatches=self.random_mini_batches(X_train,Y_train,minibatch_size,seed)
        
        for minibatch in minibatches:
          
          # Set up the current minibatch
          (minibatch_X,minibatch_Y)=minibatch
          sess.run(optimizer,
                   feed_dict={X:minibatch_X,Y:minibatch_Y})
          
        
        epoch_cost=cost.eval({X:X_train,Y:Y_train})
        
        if print_cost == True and epoch % 100==0:
          print("Cost after epoch %i: %f" %(epoch,epoch_cost))
        if print_cost == True and epoch % 5 ==0:
          costs.append(epoch_cost)
          parametersItr.append(sess.run(parameters))
        
      print("Final Cost: " +str(costs[-1]))
      parameters=sess.run(parameters)
      print("Parameters have been trained!")
      
      #predictions_train=self.forward_propagation_for_prediction(X_train, parameters)
      #print(predictions_train)
      
      #tf_ZL=sess.run(ZL,feed_dict={X:X_train,Y:Y_train})
      #print(tf_ZL)
      #predictions_test=self.forward_propagation_for_prediction(X_test,parameters)
      
      #assert (predictions_train.shape == Y_train.shape)
      #assert (predictions_test.shape == Y_test.shape)
            
      #train_msefp = np.mean((predictions_train-Y_train)**2)
      #print('Training Error using forward_prop: ' + str(train_msefp))
      #test_mse=np.mean((predictions_test-Y_test)**2)
      
      #train_mse=sess.run(cost,feed_dict={X:X_train,Y:Y_train})/m
      train_mse=cost.eval({X:X_train,Y:Y_train})
      test_mse=cost.eval({X:X_test,Y:Y_test})
      #test_mse=np.mean((predictions_test-Y_test)**2)

      
      print("Training Error: " + str(train_mse))
      print("Test Error: " + str(test_mse))
      
      
      # Arranging the information which octave will need for 
      # computing output. 
      data={'parameters':parameters,
            'parametersItr':parametersItr,
            'train_mse':train_mse,
            'test_mse':test_mse,
            'cost':costs,
            'normalize_data':self.options['normalize_data'],
            'layer_dims': self.options['layer_dims'],
            'activation': self.options['activation'],
            'cache': self.cache
            }
       
    return data
