import numpy as np

def make_sequences(features, targets, sequence_length):
    """Create ordered sequences; labels refer to the time immediately after each input window."""
    x=np.asarray(features,float); y=np.asarray(targets,int)
    return np.asarray([x[i-sequence_length:i] for i in range(sequence_length,len(x))]),y[sequence_length:]

def train_lstm(x_train,y_train,x_validation,y_validation,epochs=50,batch_size=32):
    """Train an optional Keras LSTM with validation-only early stopping."""
    try: import tensorflow as tf
    except ImportError: return None,"Skipped: TensorFlow is not installed for this Python environment."
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=x_train.shape[1:]),tf.keras.layers.LSTM(64,dropout=.2),tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dense(1,activation="sigmoid")])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.Recall(name="recall"),"accuracy"])
    stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=7,restore_best_weights=True)
    model.fit(x_train,y_train,validation_data=(x_validation,y_validation),epochs=epochs,batch_size=batch_size,shuffle=False,verbose=0,callbacks=[stop])
    return model,"Completed."
