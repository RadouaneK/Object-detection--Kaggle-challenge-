import numpy as np 
import tensorflow as tf 
from keras import backend as K
from utils import model_head, yolo_boxes_to_corners


def costum_loss(y_true, y_pred):

	#post process model output
	pred_xy, pred_wh, pred_confidence, pred_class_probs = model_head(y_pred)

	# retrieve y_true
	true_xy = y_true[..., :2] 
	true_wh = y_true[..., 2:4]

	pred_mins, pred_maxes = yolo_boxes_to_corners(pred_xy, pred_wh)
	true_mins, true_maxes = yolo_boxes_to_corners(true_xy, true_wh)

	intersect_mins  = K.maximum(pred_mins,  true_mins)
	intersect_maxes = K.minimum(pred_maxes, true_maxes)
	intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
	true_areas = true_wh[..., 0] * true_wh[..., 1]
	pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores  = intersect_areas/ union_areas

	# compute loss.
	responsibility_selector = y_true[..., 4]
	
	xy_diff = K.square(true_xy - pred_xy) * responsibility_selector[..., None]
	xy_loss = K.sum(xy_diff, axis=[1, 2, 3, 4])

	wh_diff = K.square(K.sqrt(true_wh) - K.sqrt(pred_wh)) * responsibility_selector[..., None]
	wh_loss = K.sum(wh_diff, axis=[1, 2, 3, 4])

	obj_diff = K.square(iou_scores - pred_confidence) * responsibility_selector
	obj_loss = K.sum(obj_diff, axis=[1, 2, 3])

	best_iou = K.max(iou_scores, axis=-1)
	no_obj_diff = K.square(0 - pred_confidence) * K.cast(best_iou < 0.6, K.dtype(best_iou))[..., None] * (1 - responsibility_selector)
	no_obj_loss = K.sum(no_obj_diff, axis=[1, 2, 3])

	clf_diff = K.square(y_true[..., 5:] - pred_class_probs) * responsibility_selector[..., None]
	clf_loss = K.sum(clf_diff, axis=[1, 2, 3, 4])

	object_coord_scale = 5
	object_conf_scale = 1
	noobject_conf_scale = 1
	object_class_scale = 1


	total_loss =  object_coord_scale * (xy_loss + wh_loss) + \
		object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
		object_class_scale * clf_loss

	return total_loss

