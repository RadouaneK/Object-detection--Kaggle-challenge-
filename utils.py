from keras import backend as K
import numpy as np 

anchors = np.array([[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

def model_head(y_pred, anchors = anchors):

	anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, len(anchors), 2])

	conv_dims = K.shape(y_pred)[1:3]

	conv_height_index = K.arange(0, stop=conv_dims[0])
	conv_width_index = K.arange(0, stop=conv_dims[1])

	conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
	conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
	conv_width_index = K.flatten(K.transpose(conv_width_index))

	conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
	conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
	conv_index = K.cast(conv_index, K.dtype(y_pred))

	conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(y_pred))

	box_xy = (K.sigmoid(y_pred[..., :2]) + conv_index) / conv_dims 
	box_wh = K.exp(y_pred[..., 2:4]) * anchors_tensor / conv_dims
	box_confidence = K.sigmoid(y_pred[..., 4])
	box_class_probs = K.softmax(y_pred[..., 5:])

	return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return box_mins, box_maxes



