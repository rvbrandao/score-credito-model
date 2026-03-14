from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.config import LOGISTIC_MAX_ITER
from training.config import RANDOM_STATE


def build_training_pipeline() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_STATE,
        ),
    )
