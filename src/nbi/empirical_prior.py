import numpy as np

class EmpiricalPrior:
    def __init__(self, lookup_table, priors=None, random_state=None):
        self.lookup_table = np.asarray(lookup_table)
        self.n_samples = self.lookup_table.shape[0]
        self.priors = priors
        self.rng = np.random.default_rng(random_state)

    def rvs(self, size=1, replace=True):
        if not replace and size > self.n_samples:
            raise ValueError(
                f"Requested {size} samples without replacement, but "
                f"lookup_table only has {self.n_samples} entries."
            )
        idx = self.rng.choice(self.n_samples, size=size, replace=replace)
        return self.lookup_table[idx]

    def logpdf(self, params):
        if self.priors is None:
            raise NotImplementedError(
                "logp not implemented: provide a prior model or override this method."
            )
        return self.priors.logpdf(params)
