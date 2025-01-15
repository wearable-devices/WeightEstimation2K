import keras
# import keras.ops as K
# custom metric to track sigma
class SigmaMetric(keras.metrics.Metric):
    def __init__(self, name='sigma_value', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        sigma = y_pred[:, 1]  # Extract sigma from gaussian output
        batch_mean = K.mean(sigma)
        self.total.assign_add(batch_mean)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
