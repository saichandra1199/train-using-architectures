import tensorflow as tf
from tensorflow import keras
import math
import os
import architectures
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

img_width = 224
img_height = 224
train_data_dir = '/home/vishwam/mountpoint2/Maintenance/occlusion/9aug_face_occlusion/train'
valid_data_dir = '/home/vishwam/mountpoint2/Maintenance/occlusion/9aug_face_occlusion/validation'
model_path = "architectures.py"
#filepath="gender_6f_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
epochs=30
steps_per_epoch =350
validation_steps =2800
loss='binary_crossentropy'
#def los(actual_l,pridicted_l):
	#loss=keras.losses.binary_crossentropy(actual_l, pridicted_l, from_logits=True)
	#d=math.subtract(loss,0.3,name=None)
	#loss=tf.keras.backend.maximum(0.0,d)
	#print_op = tf.print(loss, output_stream = 'file://loss.txt',summarize=-1)
	#with tf.control_dependencies([print_op]):
		#return K.identity(loss)
	#return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Blur','NonBlur'],
											   class_mode='binary',
											   batch_size=32,interpolation='lanczos')

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Blur','NonBlur'],
											   class_mode='binary',
											   batch_size=1,interpolation='lanczos')


# step-2 : build model

#model =tf.keras.models.load_model(model_path)
model=architectures.MobileNetv1()
#model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='max', baseline=None, restore_b$
#callback_list = [checkpoint]


model.compile(loss='binary_crossentropy',optimizer='Adam', metrics=['binary_accuracy'])

print('model complied!!')

print('started training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=validation_generator,validation_steps=validation_steps)

print('training finished!!')

print('saving weights to h5')

model.save('DE_blur_v1_15.h5',include_optimizer=False)

print('all weights saved successfully !!')
#models.load_weights('qqqqqqqsimple_CNN.h5')
#from tensorflow.contrib.session_bundle import exporter

#export_path = "tf_serving/" # where to save the exported graph
#export_version = 1 # version number (integer)
#
#saver = tf.train.Saver(sharded=True)
#model_exporter = exporter.Exporter(saver)
#signature = exporter.classification_signature(input_tensor=model.input,
#                                              scores_tensor=model.output)
#print(model.input)
#print(model.output)
#model_exporter.init(sess.graph.as_graph_def(),
#                    default_graph_signature=signature)
#model_exporter.export(export_path, tf.constant(export_version), sess)
#path_h5='without_motion_without_kushi_blurriness'+'/'
#path = 'tf_serving_blur_faces_v_4_0_Nima_Technical_BS_16/'
#model.save_model('without_motion_without_kushi_blurriness.h5',path_h5,include_optimizer=False,save_format='h5')
#keras.experimental.export_saved_model(model, path)

#config = model.get_config()
#model.save_weights('checkpoints_blur_faces_v_4_0_Nima_Technical_BS_16/', save_format='tf')

#tf.contrib.saved_model.save_keras_model(model, path)

