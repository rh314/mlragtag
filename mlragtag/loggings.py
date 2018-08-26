import psycopg2


class MovingAverage():
    def __init__(self, tau):
        self.inv_tau = 1. / tau
        self.averaged = []
        self.raw = []

    def add(self, new_val):
        if isinstance(new_val, (int, float)):
            pass
        elif isinstance(new_val, list):
            new_val = np.array(new_val)
        elif isinstance(new_val, np.ndarray) and new_val.ndim > 1:
            new_val = new_val.squeeze()
            if new_val.ndim > 1:
                raise Exception('new_val must be squueze-able')
        self.raw.append(new_val)
        if len(self.averaged) > 0:
            prev = self.averaged[-1]
        else:
            prev = new_val
        new_avg = prev + (new_val - prev) * self.inv_tau
        self.averaged.append(new_avg)
        return new_avg

    def numpy(self):
        return np.array(self.averaged)

    @classmethod
    def from_existing(klass, old_lst, tau):
        obj = MovingAverage(tau)
        for item in old_lst:
            obj.add(item)
        return obj
