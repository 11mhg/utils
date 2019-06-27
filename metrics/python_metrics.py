import numpy as np




def gIOU(pred_box, ground_box):
    x_pred_hat_1 = min(pred_box[0],pred_box[2])
    x_pred_hat_2 = max(pred_box[0],pred_box[2])
    y_pred_hat_1 = min(pred_box[1],pred_box[3])
    y_pred_hat_2 = max(pred_box[1],pred_box[3])

    ground_area = (ground_box[2]-ground_box[0])*(ground_box[3]-ground_box[1])
    pred_area = (x_pred_hat_2-x_pred_hat_1)*(y_pred_hat_2-y_pred_hat_1)

    x1_inter = max(x_pred_hat_1,ground_box[0])
    x2_inter = min(x_pred_hat_2,ground_box[2])
    y1_inter = max(y_pred_hat_1,ground_box[1])
    y2_inter = min(y_pred_hat_2,ground_box[3])

    if x2_inter > x1_inter and y2_inter > y1_inter:
        inter = (x2_inter - x1_inter)*(y2_inter-y1_inter)
    else:
        inter=0

    x1_c = min(x_pred_hat_1,ground_box[0])
    x2_c = max(x_pred_hat_2,ground_box[2])
    y1_c = min(y_pred_hat_1,ground_box[1])
    y2_c = max(y_pred_hat_2,ground_box[3])

    c_area = (x2_c-x1_c)*(y2_c-y1_c)

    union = pred_area+ground_area - inter
    iou = inter/union
    giou = iou - ((c_area-union)/c_area)
    return iou,giou

def loss_iou(pred_box,ground_box):
    iou, giou = gIOU(pred_box,ground_box)
    return 1-iou, 1-giou

