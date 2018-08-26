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

    def append(self, x):
        if not isinstance(x, (float, int)):
            raise TypeError
        tstamp = time.time()
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                'INSERT INTO %s VALUES(%f, %f)' %
                (self.uuid, tstamp, x))

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
            res = [_[0] for _ in cur.fetchall()]
        return res


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
