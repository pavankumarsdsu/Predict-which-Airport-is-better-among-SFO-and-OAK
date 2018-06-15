from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
#from keras.backend.cntk_backend import dropout

class FeedForward:
    def __init__(self, x_train, x_test, y_train, y_test, input_length):
        
        # define the architecture of the network
        model = Sequential()
        model.add(Dense(input_length, input_dim=input_length, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(input_length, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation("softmax"))
        
        model.compile(optimizer = "Adam",  
                      loss = "binary_crossentropy",
                      metrics = ['categorical_accuracy'])
        
        model.fit(x_train, y_train, epochs=50, batch_size=100)
        
        #Calculate accuracy of model
        (loss, accuracy) = model.evaluate(x_test, y_test,
        batch_size=100, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
        accuracy * 100))