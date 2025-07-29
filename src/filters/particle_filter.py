class ParticleFilter:
    def __init__(self, n_particles):
        self.n_particles = n_particles
    def fit(self, data):
        print(f"Fitting Particle Filter with {self.n_particles} particles on data of shape {data.shape}") 