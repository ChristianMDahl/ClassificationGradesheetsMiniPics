#---------------------------------------------------------
# Classification of mini_pics from sweedish grading sheets
#---------------------------------------------------------

import tensorflow as tf
#Re_estimate = 1: New model is trained
#Re_estimate = 0: Most recently trianed model is loaded
##################
Re_estimate = 0 ##
##################

data_train = tf.contrib.keras.preprocessing.image.DirectoryIterator(directory='Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\goodsmall\\gradessmall\\train',
                                                                    image_data_generator=tf.contrib.keras.preprocessing.image.ImageDataGenerator(rescale=1./255),
                                                                    color_mode = 'grayscale',
                                                                    class_mode ='categorical',
                                                                    target_size=(28, 28))

# To make this work efficient we need a directory - similar to gradesmall - where we have the data for validation/test
data_test = tf.contrib.keras.preprocessing.image.DirectoryIterator(directory='Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\goodsmall\\gradessmall\\test',
                                                                   image_data_generator=tf.contrib.keras.preprocessing.image.ImageDataGenerator(rescale=1./255),
                                                                   color_mode = 'grayscale',
                                                                   class_mode ='categorical',
                                                                   target_size=(28, 28))
if Re_estimate == 1:
    model = tf.contrib.keras.models.Sequential()
    model.add(tf.contrib.keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=data_train.image_shape))
    model.add(tf.contrib.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.contrib.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.contrib.keras.layers.Dropout(0.25))
    model.add(tf.contrib.keras.layers.Flatten())
    model.add(tf.contrib.keras.layers.Dense(128, activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.5))
    model.add(tf.contrib.keras.layers.Dense(data_train.num_class, activation='softmax'))
    
    model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy,
                  optimizer=tf.contrib.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    # with tf.device('/gpu:1'):   #Unfortunately this does not work...probably need to install tensorflow with the GPU open
    model.fit_generator(data_train,steps_per_epoch=200,epochs=100,validation_data=data_test,validation_steps= data_test.batch_size)
    
    score = model.evaluate_generator(data_train,steps=data_train.batch_size)                                                       
    
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    
    score = model.evaluate_generator(data_test,steps=data_test.batch_size)                                                       
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
#serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

if Re_estimate == 0: 
# load YAML and create model
    yaml_file           = open('model.yaml', 'r')
    loaded_model_yaml   = yaml_file.read()
    yaml_file.close()
    loaded_model        = tf.contrib.keras.models.model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")     
    # evaluate loaded model on test data
    loaded_model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy,
                  optimizer=tf.contrib.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    score = loaded_model.evaluate_generator(data_test,steps=data_test.batch_size)                                                       
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

