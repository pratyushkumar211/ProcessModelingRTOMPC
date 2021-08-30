import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

class CustomLoss(tf.keras.losses.Loss):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        """ Write the call function. """
        
        # Custom MSE.
        y_pred_mse = y_pred[..., 0:1] # Relevant for mse
        
        y_pred_1 = y_pred[..., 1:2]
        y_pred_2 = y_pred[..., 2:3]

        cost = tf.math.reduce_mean(tf.square((y_true - y_pred)))
        cost += tf.math.reduce_mean(tf.square(10*(y_pred_1 - y_pred_2)))

        # Return.
        return cost

def create_model(fNdims, xyscales):
    """ Function to create the model. """

    # Symbolic placeholder for NN input.
    nnInput = tf.keras.Input(shape=(fNdims[0], ))

    # Create layers. 
    layers = []
    for dim in fNdims[1:-1]:
        layers += [tf.keras.layers.Dense(dim, activation='relu')]
    layers += [tf.keras.layers.Dense(fNdims[-1])]

    # Get output symbolically.
    nnOutput = nnInput
    for layer in layers:
        nnOutput = layer(nnOutput)
    
    # Construct a model.
    model = tf.keras.Model(inputs=nnInput, outputs=nnOutput)

    # Compile model. 
    ystd = xyscales['y'][1]
    loss = CustomLoss()
    model.compile(optimizer='adam', loss=loss)

    # Return a model object.
    return model

def train_model(model, epochs, batch_size, traindata, 
                stdout_filename, ckpt_path):
    """ Function to train the model. """

    # Create stdout.
    sys.stdout = open(stdout_filename, 'w')

    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    # Call the fit method to train.
    history = model.fit(x = traindata['x'], 
              y = traindata['y'], 
              validation_split = 0.2, 
              epochs = epochs, batch_size = batch_size,
            callbacks = [checkpoint_callback])

    # Get losses over epochs for plotting.
    train_loss = np.array(history.history['loss'])
    trainval_loss = np.array(history.history['val_loss'])

    # Return trained model. 
    return model, train_loss, trainval_loss

def get_val_loss(model, testdata, ckpt_path):
    """ First load model weights, and get predictions. """

    # Load weights. 
    model.load_weights(ckpt_path)

    # Get the validation metrics. 
    val_loss = model.evaluate(x=testdata['x'], y=testdata['y'])    

    # Return.
    return val_loss

def get_dataset():
    """ Get the train and test boston dataset. """

    # First get training and testing data.
    train_test_data = tf.keras.datasets.boston_housing.load_data(test_split=0.2)
    (xtrain, ytrain), (xtest, ytest) = train_test_data

    # Get scaling for training and testing data.
    xmean = np.mean(xtrain, axis=0)
    xstd = np.std(xtrain, axis=0)
    ymean = np.mean(ytrain, axis=0)
    ystd = np.std(ytrain, axis=0)

    # Get scaled train and test data data.
    traindata = {}
    traindata['x'] = (xtrain - xmean)/xstd
    traindata['y'] = (ytrain - ymean)/ystd

    testdata = {}
    testdata['x'] = (xtest - xmean)/xstd
    testdata['y'] = (ytest - ymean)/ystd

    # Scale dict.
    xyscales = dict(x=(xmean, xstd), y=(ymean, ystd))

    # Return.
    return traindata, testdata, xyscales


# Get data set.
traindata, testdata, xyscales = get_dataset()

# Get model. 
fNdims = [13, 16, 3]
model = create_model(fNdims, xyscales)

# Train model. 
stdout_filename = 'regression.txt'
ckpt_path = 'regression.ckpt'
epochs = 400
batch_size = 32
model, train_loss, trainval_loss = train_model(model, epochs, 
                                               batch_size, traindata,
                                               stdout_filename, ckpt_path)

# Compute metric.
val_loss = get_val_loss(model, testdata, ckpt_path)

# Print validation loss. 
print("Validation loss: " + str(val_loss))

# Plot the losses during training. 
plt.figure(figsize=(5, 4))
plt.plot(train_loss, label='Training data loss')
plt.plot(trainval_loss, label='Buffer data loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss', rotation=False, labelpad=10)
plt.xlim([0, epochs-1])
plt.tight_layout(pad=1)
plt.savefig('regression.pdf')