import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import optimize
from math import log

def D(ts, ps):
    return sum(t*log(t/p) for t,p in zip(ts,ps))

class Upper:
    def __init__(self, w1, wq, wu, eps=1e-3, arg_gran=100):
        self.w1 = w1
        self.wq = wq
        self.wu = wu
        self.P = np.array([[w1, wq-w1], [wu-w1, 1-wq-wu+w1]])
        self.eps = eps
        self.arg_gran = arg_gran

    def zspace(self, arg):
        w1, wq, wu = self.w1, self.wq, self.wu
        r = w1 - wq*wu
        a = -1 + arg*(w1 + wu*(-1-wq+wu))/r
        b = arg + (1-arg)*(1-wq)*wq/r
        b1 = sorted([wq/b, -(1-wq)/b])
        b2 = sorted([-wu/a, (1-wu)/a])
        return (max(b1[0],b2[0]), min(b1[1],b2[1]))

    def tqtu(self, arg, z=None):
        """ Paramtic (tq,tu) at z on trade-off line with slope a.
            If z is not given, it is determined automatically. """
        if z is None:
            z = self.findz(arg)
        w1, wq, wu = self.w1, self.wq, self.wu
        r = w1 - wq*wu
        a = -1 + arg*(w1 + wu*(-1-wq+wu))/r
        b = arg + (1-arg)*(1-wq)*wq/r
        tq = -b*z + wq
        tu = a*z + wu
        return (tq, tu)

    def level(self, T):
        P = self.P
        marg_x = np.log(T.sum(axis=1)/P.sum(axis=1))
        marg_y = np.log(T.sum(axis=0)/P.sum(axis=0))
        tau = np.log(T/P)
        marg_x[1] *= -1
        marg_y[1] *= -1
        den = marg_x[0] * marg_y[0] * (tau[0,0]-tau[0,1]) * (tau[0,0]-tau[1,0])
        marg_x = np.flip(marg_x)
        marg_y = np.flip(marg_y)
        return marg_x @ tau @ marg_y / den

    def t1f(self, tq, tu):
        P = self.P
        a = P[0,0]*P[1,1]
        b = P[0,1]*P[1,0]
        rs = np.roots([a-b, -a*(tq+tu)-b*(1-tq-tu), a*tq*tu])
        return rs[1]

    def f(self, arg, z):
        P = self.P
        tq, tu = self.tqtu(arg, z)
        assert ((0 <= tq <= 1) and (0 <= tu <= 1))
        t1 = self.t1f(tq, tu)
        T = np.array([[t1, tq-t1], [tu-t1, 1-tq-tu+t1]])
        return self.level(T)

    def findz(self, a):
        z_min, z_max = self.zspace(a)
        fp = lambda z: self.f(a, z)
        for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            if eps < z_max-eps and np.sign(fp(eps)) != np.sign(fp(z_max-eps)):
                return optimize.bisect(fp, eps, z_max-eps)
            elif z_min+eps < -eps and np.sign(fp(-eps)) != np.sign(fp(z_min+eps)):
                return optimize.bisect(fp, z_min+eps, -eps)
            else:
                pass
        print('Unable to find best', self.w1, self.wq, self.wu, 'a', a)
        #zs = np.linspace(z_min, z_max)
        #plt.plot(zs, list(map(fp,zs)))
        #plt.show()
        return 0
        # There should always be a root at z=0, but the other root
        # may be either at z negative or positive.
        # Example with negative z: (.3, .4, .6)

    def rho(self, a):
        z = self.findz(a)
        tq, tu = self.tqtu(a, z)
        t1 = self.t1f(tq, tu)
        w1, wq, wu = self.w1, self.wq, self.wu
        DTP = D((t1,tq-t1,tu-t1,1-tq-tu+t1), (w1,wq-w1,wu-w1,1-wq-wu+w1))
        Dq = D((tq,1-tq),(wq,1-wq))
        Du = D((tu,1-tu),(wu,1-wu))
        return (DTP-Dq)/Du, (DTP-Du)/Du

    def balanced(self, q_weight=1):
        # Finds `a` such that rq == ru (or rq * q_weight == ru)
        eps = self.eps
        def f(a):
            rq, ru = self.rho(a)
            return rq*q_weight - ru
        a = optimize.bisect(f, eps, 1-eps)
        rq, ru = self.rho(a)
        return a, (rq+ru)/2

if __name__ == '__main__':
    #upper = Upper(.3, .4, .6)
    u = Upper(.1, .2, .3)
    print(u.t1f(.7,.8))

    eps = self.eps
    wus = np.linspace(.1+eps, .3-eps)
    plt.plot(wus, [Upper(.1,wu,.3).balanced() for wu in wus])
    plt.show()

    w1s = np.linspace(.06+eps, .2-eps)
    plt.plot(w1s, [Upper(w1,.2,.3).balanced() for w1 in w1s])
    plt.show()

    upper = Upper(.1, .2, .3)
    aas = np.linspace(1e-3,1-1e-3)
    rqrus = [upper.rho(a) for a in aas]
    plt.plot(*zip(*rqrus))
    plt.plot(np.linspace(0,1), np.linspace(0,1))
    plt.show()


