import tensorflow as tf

#load and prepare the mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train /255.0, x_test/255.0

#build tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential( [
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#for each example the model returns a vector of "logits and "logodds"
#score for each class
#logits: the vector of raw (non-normalized) predicts that a classification
#model generates; typically become the input to the softmax func
#log-odds: the logarithm of the odds of some event (inverse of the sigmoid func)
predictions = model(x_train[:1]).numpy()

#tf.nn.softmax func converts logits to probabilities
#it is possible to use softmax as the activation func for the last layer
#of the network (although discouraged)
tf.nn.softmax(predictions).numpy()

#losses. ... takes a vector of logits and a True index and return a
#scalar loss
#the loss is equal to the negative log probability of the true class:
#=0 if the model is sure of the correct class
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#this untrained model gives probabilities close to random(1/10 for each class),
#so the initial loss should be close to -tf.log(1/10) ~=2.3
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#the Model.fit method adjusts the model parameters to minimize the loss:
model.fit(x_train, y_train, epochs=5)

#the Mode.evaluate method checks the models performance, usually on a
#CV or Test set
#the image classifier SHOULD BE now trained to ~98% accuracy on this dataset.
model.evaluate(x_test, y_test, verbose=2)

#if you want your model to return a probability, you can wrap the trained model,
#and attach the softmax to it:
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

