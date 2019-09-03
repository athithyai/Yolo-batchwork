# hello_world +__

from train import get_classes
from train import get_anchors
from train import get_classes
import sys
import numpy as np
from keras.optimizers import Adam
from train import data_generator_wrapper


base_data_path = "/dbfs/FileStore/tables/gtsdb/"

# append to python path
sys.path.append(base_data_path + "yolo")


#fix some paths
log_dir = "logs/000/"
base_data_path = "/dbfs/FileStore/tables/gtsdb/"
annotation_file_training = base_data_path + "data/trainlist.txt"
annotation_file_validation = base_data_path + "data/vallist.txt"
classes_path = base_data_path + "data/gtsdb_classes.txt"
anchor_path = base_data_path + 'yolo/model_data/yolov3_tiny_anchors.txt'

class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchor_path)

input_shape = (416, 416)
from train import create_tiny_model
print(base_data_path + 'yolo/model_data/yolov3_tiny.h5')
model = create_tiny_model(input_shape, anchors, num_classes,
    freeze_body=2, weights_path=base_data_path + 'yolo/model_data/yolov3_tiny.h5')

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


training_lines = []
validation_lines = []
with open(annotation_file_training) as f:
  training_lines = f.readlines()
with open(annotation_file_validation) as f:
  validation_lines = f.readlines()

  
np.random.seed(42)
np.random.shuffle(training_lines)

num_train = len(training_lines)
num_val = len(validation_lines)


model.compile(optimizer=Adam(lr=1e-2), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

	
batch_size = 8
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
model.fit_generator(data_generator_wrapper(training_lines, batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(validation_lines, batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=1,
        initial_epoch=0,
        callbacks=[logging, checkpoint])
model.save_weights(log_dir + 'trained_weights_stage_1.h5')
