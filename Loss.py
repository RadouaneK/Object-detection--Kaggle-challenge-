import numpy as np 
from keras import backend as K
from utils import model_head, yolo_boxes_to_corners


def costum_loss(y_true, y_pred):

	#post process model output
	box_xy, box_wh, box_confidence, box_class_probs = model_head(y_pred, K.shape(y_pred)[3])

	# retrieve y_true
	true_box_xy = y_true[..., :2] 
    true_box_wh = y_true[..., 2:4]

    pred_box_mins, pred_box_maxes = yolo_boxes_to_corners(box_xy, box_wh)
    true_box_mins, true_box_maxes = yolo_boxes_to_corners(true_box_xy, true_box_wh)

	intersect_mins  = K.maximum(pred_box_mins,  true_box_mins)
	intersect_maxes = K.minimum(pred_box_maxes, true_box_maxes)
	intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
	true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
	pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores  = intersect_areas/ union_areas

	# Best IOU scores.
	best_ious = K.max(iou_scores, axis=4)  
    best_ious = K.expand_dims(best_ious)

    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))


with tf.Session() as sess:
	print(sess.run(cell_grid).shape)
	print(sess.run(conv_index).shape)