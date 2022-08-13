"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""

from lightgbm import LGBMRegressor

def estimator_fn():
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #from sklearn.linear_model import SGDRegressor
    #return SGDRegressor(random_state=42)
  
    return LGBMRegressor(
              colsample_bytree=0.7201078529232644,
              lambda_l1=0.7730308267776886,
              lambda_l2=5.890400206116792,
              learning_rate=0.03469799540212862,
              max_bin=482,
              max_depth=12,
              min_child_samples=58,
              n_estimators=623,
              num_leaves=36,
              subsample=0.7203410786015857,
              random_state=77049990,
            )
  
    
