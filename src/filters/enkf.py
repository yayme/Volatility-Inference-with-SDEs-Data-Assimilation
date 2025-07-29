class EnKF:
    def __init__(self, ensemble_size):
        self.ensemble_size = ensemble_size
    def fit(self, data):
        print(f"Fitting EnKF with {self.ensemble_size} ensemble members on data of shape {data.shape}") 