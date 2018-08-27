import psycopg2
import time
import struct
import numpy as np


db_holder = [None]


def get_db():
    '''
    Initialize db if not connected yet.
    PS: Need to setup postgres first:
        cd ~/postgres/bin
        ./createuser -e mlragtag
        ./createdb -e -Omlragtag mlragtag_db
    '''
    if db_holder[0] is None:
        db = psycopg2.connect(
            "dbname='mlragtag_db' user='mlragtag' host='localhost'")
        db_holder[0] = db
        with db.cursor() as cur:
            cur.execute("SET synchronous_commit = 'on'")
    return db_holder[0]


def new_uuid():
    return struct.pack('d', time.time()).hex()


class List():
    # PS: Per row fastest insert is if you extend with about 50 items at a
    # time.
    def __init__(self):
        self.uuid = 't' + new_uuid()
        db = get_db()
        with db.cursor() as cur:
            cur.execute('''
                CREATE TABLE %s(tstamp REAL, value REAL)
                ''' % self.uuid)
            db.commit()
        self.queue = []

    def append(self, x):
        if not isinstance(x, (float, int)):
            raise TypeError
        tstamp = time.time()
        queue = self.queue
        self.last = x + 0
        queue.append((tstamp, self.last))
        if len(queue) >= 50:
            db = get_db()
            cmd = 'INSERT INTO %s VALUES(%%s, %%s)' % self.uuid
            cmd = ';'.join(((cmd,) * len(queue)))
            queue_flat = []
            for pair in queue:
                queue_flat.extend(pair)
            with db.cursor() as cur:
                cur.execute(cmd, queue_flat)
            del queue[:]

    def extend(self, x):
        rowdata = []
        tstamp = time.time()
        for xi in x:
            rowdata.extend((tstamp, xi))
            if not isinstance(xi, (float, int)):
                raise TypeError
        db = get_db()
        cmd = 'INSERT INTO %s VALUES(%%s, %%s)' % self.uuid
        cmd = ';'.join(((cmd,) * len(x)))
        with db.cursor() as cur:
            cur.execute(cmd, rowdata)

    def get_values(self):
        db = get_db()
        with db.cursor() as cur:
            cur.execute('SELECT value FROM %s' % self.uuid)
            fetched = cur.fetchall()
            res = [_[0] for _ in fetched]
        res.extend([_[1] for _ in self.queue])
        return res


class MovingAverage():
    def __init__(self, tau, size=None):
        self.inv_tau = 1. / tau
        self.size = size
        if size is None:
            self.averaged = List()
            self.raw = List()
        else:
            self.wrapped = [MovingAverage(tau) for _ in range(size)]
        self.last = None

    def add(self, new_val):
        if self.size is not None:
            assert len(new_val) == self.size
            for item in new_val:
                assert isinstance(item, (int, float))
            next_vals = []
            for k in range(self.size):
                next_vals.append(
                    self.wrapped[k].add(new_val[k]))
            self.last = next_vals
            return next_vals
        if isinstance(new_val, (int, float)):
            pass
        else:
            raise TypeError
        self.raw.append(new_val)
        if self.last is None:
            prev = new_val
        else:
            prev = self.last
        next_val = prev + (new_val - prev) * self.inv_tau
        self.averaged.append(next_val)
        self.last = next_val
        return next_val

    def numpy(self):
        if self.size is None:
            return np.array(self.averaged.get_values())
        else:
            values = []
            for item in self.wrapped:
                values.append(item.averaged.get_values())
            return np.array(values).T

    @classmethod
    def from_existing(klass, old_lst, tau):
        raise Exception('TODO')
        obj = MovingAverage(tau)
        for item in old_lst:
            obj.add(item)
        return obj
