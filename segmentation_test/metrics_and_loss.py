# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_metrics_and_loss.ipynb.

# %% auto 0
__all__ = ['dice_coef', 'tversky_index', 'FNR', 'FDR', 'dice_coef_loss', 'dice_loss', 'tversky_focal_loss']

# %% ../nbs/04_metrics_and_loss.ipynb 3
import tensorflow as tf

# %% ../nbs/04_metrics_and_loss.ipynb 4
def dice_coef(y_true, y_pred, smooth=0):        
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection =  tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / ( tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice


# %% ../nbs/04_metrics_and_loss.ipynb 5
def tversky_index(alpha=0.5, beta=0.5, gamma=1.5,name=None):
    def TI(y_true, y_pred):
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)
    
        true_positives = tf.keras.backend.sum(y_true * y_pred)
        false_positives = tf.keras.backend.sum(y_pred * (1 - y_true))
        false_negatives = tf.keras.backend.sum((1 - y_pred) * y_true)

        numerator = true_positives + 1e-7
        denominator = true_positives + alpha * false_negatives + beta * false_positives + 1e-7
        tversky_index = numerator / denominator
        return tversky_index
    if name is None:
        TI.__name__ = "TI_%.02f_%.02f_%.02f" % (alpha, beta, gamma)
    else:
        TI.__name__ = name
    return TI


class FNR(tf.keras.metrics.Recall):
    """! FNR - False negative rate metric
             FN / (FN + TP)
    """
    def __init__(self, thresholds=0.5, name="fnr_escapee", **kwargs):
        super().__init__(name=name, thresholds=thresholds, **kwargs)
        
    def result(self):
        return 1.0 - super().result()
    
    
class FDR(tf.keras.metrics.Precision):
    """! FDR - False discovery rate metric
            FP / (FP + TP)
    """
    def __init__(self, thresholds=0.5, name="fdr_overreject", **kwargs):
        super().__init__(name=name, thresholds=thresholds, **kwargs)
        
    def result(self):
        return 1.0 - super().result()
        
    
#
# Loss Functions
#

def dice_coef_loss(y_true, y_pred):
    #return 1 - tf.keras.backend.log(dice_coef(y_true, y_pred))
    return 1 - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred)

def tversky_focal_loss(alpha=0.5, beta=0.5, gamma=1.5, name=None):
    def TFL(y_true, y_pred):
        ti = tversky_index(alpha, beta, gamma)
        return tf.keras.backend.pow(1 - ti(y_true, y_pred), gamma)
    if name is None:
        TFL.__name__ = "TFL_%.02f_%.02f_%.02f" % (alpha, beta, gamma)
    else:
        TFL.__name__ = name
    
    return TFL
