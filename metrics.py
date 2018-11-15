#encoding: utf-8
from keras import backend as K
import tensorflow as tf
import numpy as np
from skimage import morphology as m
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = K.sum(y_true_f * y_pred_f) + 1.0
    union = K.sum(y_true_f) + K.sum(y_pred_f) + 1.0
    dice = 2. * intersection / (union)

    return dice


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1.0 - dice


def tversky_focal_loss(alpha=0.5, beta=2, gamma=3):

    def warped_loss(y_true, y_pred):
        # tversky loss
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        ones = K.ones(K.shape(y_true))
        prob1 = y_pred
        prob0 = ones - y_pred
        mask1 = y_true
        mask0 = ones - y_true
        intersection = K.sum(prob1 * mask1)
        weighted_union = intersection + alpha * \
            K.sum(prob0 * mask1) + (1.0 - alpha) * \
            K.sum(prob1 * mask0) + K.epsilon()
        t_loss = K.sum(intersection / weighted_union)
        # focal loss
        focal_t = -beta * K.pow(1. - t_loss, gamma) * \
            K.log(t_loss + K.epsilon())

        return focal_t

    return warped_loss


def tversky_index(y_true, y_pred, alpha=0.75):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    FN_area = (1.0 - y_pred_f) * y_true_f
    FP_area = y_pred_f * (1.0 - y_true_f)
    weighted_union = intersection + alpha * K.sum(FN_area) +\
                     (1.0 - alpha) * K.sum(FP_area) + K.epsilon()

    return intersection / weighted_union


def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f*y_pred_f) + K.epsilon()

    return intersection / union


def iou_loss(y_true, y_pred):
    """jaccard index loss"""
    return 1.0 - jaccard_index(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """ternaus net
    """
    bce = binary_crossentropy(y_true, y_pred)

    return bce + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred, weight=1.0):
    """ternaus net
    """
    bce = binary_crossentropy(y_true, y_pred)

    return weight*bce - K.log(1.0 - dice_loss(y_true, y_pred))


def bce_logjaccard_loss(y_true, y_pred):
    """ternaus net
    """
    jaccard = jaccard_index(y_true, y_pred)
    bce = binary_crossentropy(y_true, y_pred)
    return bce - K.log(jaccard)


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) +
                     K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def bce_with_tv_loss(y_true, y_pred):
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    tversky = tversky_index(y_true, y_pred)

    return bce - K.log(tversky + K.epsilon())


def mean_iou(y_true, y_pred):
    """F2 loss"""
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)

    return K.mean(K.stack(prec), axis=0)


def calc_IoU(A, B):
    AorB = np.logical_or(A, B).astype('int')
    AandB = np.logical_and(A, B).astype('int')
    IoU = AandB.sum() / AorB.sum()
    return IoU


def calc_IoU_vector(A, B):
    """计算不同阈值下的IOU"""
    score_vector = []
    IoU = calc_IoU(A, B)
    for threshold in np.arange(0.5, 1, 0.05):
        score = int(IoU > threshold)
        score_vector.append(score)
    return score_vector


def calc_IoU_tensor(masks_true, masks_pred):
    true_mask_num = masks_true.shape[0]
    pred_mask_num = masks_pred.shape[0]
    score_tensor = np.zeros((true_mask_num, pred_mask_num, 10))
    for true_i in range(true_mask_num):
        for pred_i in range(pred_mask_num):
            true_mask = masks_true[true_i]
            pred_mask = masks_pred[pred_i]
            score_vector = calc_IoU_vector(true_mask, pred_mask)
            score_tensor[true_i, pred_i, :] = score_vector

    return score_tensor


def calc_F2_per_one_threshold(score_matrix):
    tp = np.sum(score_matrix.sum(axis=1) > 0)
    fp = np.sum(score_matrix.sum(axis=1) == 0)
    fn = np.sum(score_matrix.sum(axis=0) == 0)
    F2 = (5*tp) / ((5*tp) + fp + (4*fn))

    return F2


def calc_score_one_image(mask_true, mask_pred):
    mask_true = mask_true.reshape(768, 768)
    mask_pred = mask_pred.reshape(768, 768)
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        score = 1
    elif mask_true.sum() == 0 and mask_pred.sum() != 0:
        score = 0
    elif mask_true.sum() != 0 and mask_pred.sum() == 0:
        score = 0
    else:
        mask_label_true = m.label(mask_true)
        mask_label_pred = m.label(mask_pred)
        c_true = np.max(mask_label_true)
        c_pred = np.max(mask_label_pred)
        tmp = []
        for k in range(c_true):
            tmp.append(mask_label_true == k+1)
        masks_true = np.stack(tmp, axis=0)
        tmp = []
        for k in range(c_pred):
            tmp.append(mask_label_pred == k+1)
        masks_pred = np.stack(tmp, axis=0)
        score_tensor = calc_IoU_tensor(masks_true, masks_pred)
        F2_t = []
        for i in range(10):
            F2 = calc_F2_per_one_threshold(score_tensor[:, :, i])
            F2_t.append(F2)
        score = np.mean(F2_t)

    return score


def calc_score_all_image(batch_mask_true, batch_mask_pred, threshold=0.5):
    num = batch_mask_true.shape[0]
    tmp = batch_mask_pred > threshold
    batch_mask_pred = tmp.astype('int')
    scores = list()
    for i in range(num):
        score = calc_score_one_image(batch_mask_true[i], batch_mask_pred[i])
        scores.append(score)

    return np.mean(scores)


# RGB图像增加颜色增强
# HSV变换
# 当目标很小时，可以采用crop的方式
# 为了降低模型的预测方差，将训练集采用较差验证的方式，预测四份模型，最后整合
# 每个模型的预测结果
# epoch < 30时，如果val loss不在下降: lr / 2
# 循环lr, 周期为两个epochs， good idear
# pseudo-labeling
# linknet
# 在label中设置加权数值，根据距离边界远近， 边界值权重是中心的3倍
# 将数据增强分为两个阶段，stage1 重量级增强， stage2 轻量级增强

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)

    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:

        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        losses.set_shape((None,))
        loss = tf.reduce_mean(losses)

    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        # 将label转换为1, -1
        signs = 2. * labelsf - 1.
        # 前景和背景误差
        errors = 1. - logits * tf.stop_gradient(signs)
        # 将预测像素的误差排序
        errors_sorted, perm = tf.nn.top_k(errors, 
                                          k=tf.shape(errors)[0],
                                          name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------
def lovasz_softmax(probas, labels, classes='all', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore, order), classes=classes)

    return loss


def lovasz_softmax_flat(probas, labels, classes='all'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = 1
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        # foreground for class c
        fg = tf.cast(tf.equal(labels, c), probas.dtype)
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(
            errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(
                grad), 1, name="loss_class_{}".format(c))
        )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)

    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = 1
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')

    return vprobas, vlabels


def focal_loss(y_true, y_pred):
    gamma = 0.75
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, K.epsilon(), 1.0 - K.epsilon())
    pt_0 = K.clip(pt_0, K.epsilon(), 1.0 - K.epsilon())
    loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) -\
                  (1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    
    return loss

