import time
import math
import collections
from opt2 import Upper, D
import random
import numpy.random


def sample(w1, wq, wu):
    x = random.random()
    if x < w1: return (1,1)
    if x < wq: return (1,0)
    if x < wq+wu-w1: return (0,1)
    return (0,0)

def est(f, eps):
    a, b = 0, 0
    #while a+b >= min(a**2, b**2) / 10:
    while min(a, b) < eps**(-2):
        if f(): a += 1
        else:   b += 1
    return a/(a + b)

def mc(w1, wq, wu, D, tq, tu, sigma, U, k):
    sq, su = sigma * math.sqrt(tq*(1-tq)), sigma * math.sqrt(tu*(1-tu))
    individuals = [(0,0)]
    size = 1
    for l in range(1, k+1):
        level_size = math.ceil(D**l/size)
        size *= level_size
        new_individuals = []
        for vq, vu in individuals:
            # Using a binomial like this is more honest to what we do in the real case
            #actual = numpy.random.binomial(U, level_size/U)
            #for _ in range(actual):
            for _ in range(level_size):
                dq, du = sample(w1, wq, wu)
                vq1, vu1 = vq + dq, vu + du
                if abs(vq1 - tq*l) <= sq*l**.5 and abs(vu1 - tu*l) <= su*l**.5:
                    new_individuals.append((vq1, vu1))
        individuals = new_individuals
    return bool(individuals)

class Supermajority:
    def __init__(self, U, wq, wu, w1, n, q_weight=1):
        self.U = U # TODO: Make sure prime
        self.sigma = 1
        self.lists = collections.defaultdict(list)
        self.wq, self.wu, self.w1, self.w2 = wq, wu, w1, wq*wu
        u = Upper(w1, wq, wu)
        a, _ = u.balanced(q_weight = q_weight)
        rq, ru = u.rho(a)
        print(f'Calculated rho values of rq: {rq:.3}, ru: {ru:.3}')
        self.tq, self.tu = u.tqtu(a)
        t1 = u.t1f(self.tq, self.tu)
        tq, tu = self.tq, self.tu
        print(f'tq={tq:.3}, tu={tu:.3}')
        self.D1 = D([t1, tq-t1, tu-t1, 1-tq-tu+t1], [w1, wq-w1, wu-w1, 1-wq-wu+w1])
        dq, du = D([tq,1-tq], [wq, 1-wq]), D([tu,1-tu], [wu, 1-wu])
        self.D2 = dq + du

        self.n = n
        self.k = int(math.ceil(math.log(n) / (self.D2 - dq)))
        print(f'Branch factor: {math.exp(self.D1):.3}, Height: {self.k}')
        print('Running simulations')
        surv = est(lambda: mc(self.w1, self.wq, self.wu, math.exp(self.D1), self.tq, self.tu, self.sigma, self.U, self.k), .1)
        # TODO: Might try to optimize the tradeoff between SIGMA and surv here.
        repeats = int(math.ceil(5 / surv)) # 5 ~ log(100)
        print(f'Doing {repeats} repeats.')
        self.seeds = [random.randrange(self.U**3) for _ in range(repeats)]
        self.sa = [[random.randrange(1, self.U**3) for _ in range(self.k+1)] for _ in range(repeats)]
        self.sb = [[random.randrange(self.U**3) for _ in range(self.k+1)] for _ in range(repeats)]


    def build(self, data):
        if len(data) != self.n:
            print(f'warning: supermajority configured for n={self.n}, but building on {len(data)} points.')

        start = time.time()
        for i, point in enumerate(data):
            print(f'inserting point {i}', end='\r', flush=true)
            for rep in self.reprs(point, self.tu):
                self.lists[rep].append(point)
        print()
        nlsts = sum(len(lst) for lst in self.lists.values())
        print(f'average lists per data point: {nlsts/len(data):.3f}')
        print(f'average time per insert: {(time.time()-start)/len(data):.3f}')

    def query(self, point):
        cnt = collections.counter()
        for rep in self.reprs(point, self.tq):
            cnt[rep] += 1
            for point2 in self.lists[rep]:
                yield point2
        print(f'queried {sum(cnt.values())} buckets. {len(cnt)} unique.')

    def reprs(self, point, t):
        for seed in self.seeds:
            individuals = [(0, seed)]
            size = 1
            for l in range(1, self.k+1):
                #sig = max(self.sigma * math.sqrt(t*(1-t)*l*(1-l/self.k)), 1)
                sig = self.sigma * math.sqrt(t*(1-t)*l)
                level_size = math.ceil(math.exp(l * self.D1) / size)
                size *= level_size
                new_individuals = []
                for v, h in individuals:
                    # TODO: The paper has a way to improve the speed of this step using two-independent hash functions
                    random.seed(h)
                    for x in random.sample(range(self.U), level_size):
                        h1 = random.randrange(self.U**3)
                        v1 = v + bool(x in point)
                        if abs(v1 - t*l) <= sig:
                            new_individuals.append((v1, h1))
                individuals = new_individuals
            yield from (h for v,h in individuals)


def main():
    #U = 10**6 + 3
    
    #U = 10**4 + 7
    U = 10**3 + 9
    #n = 10**5
    n = 10**4
    w1, wq, wu = int(.1 * U), int(.2 * U), int(.2 * U)
    data = [set(random.sample(range(U), wu)) for _ in range(n-1)]
    plant = random.sample(range(U), w1)
    plant_point = set(plant + random.sample(range(U), wu-w1))
    data.append(plant_point)
    print(f'Generated {len(data)} data points')

    query = set(plant + random.sample(range(U), wq-w1))

    sm = Supermajority(U, wq/U, wu/U, w1/U, n, q_weight=1)
    print('Building data structure')
    #sm.build(data)
    sm.build([plant_point])
    print('Querying data structure')
    cnt = collections.Counter()
    best = 0, None
    for point in sm.query(query):
        inter = len(point&query)
        if inter > best[0]:
            best = (inter, point)
        cnt[inter] += 1
    print(f'Saw {sum(cnt.values())} points')
    print(cnt.most_common(10))
    print(f'Best was: {best}')

def main2():
    U = 10**3 + 9
    n = 10**4
    w1, wq, wu = int(.1 * U), int(.2 * U), int(.2 * U)
    a, b = 0, 0
    def f():
        # With this approach we actually get w1' = w1 + wq wu/(1-w1)
        # So it's kinda cheating.
        plant = random.sample(range(U), w1)
        plant_point = set(plant + random.sample(range(U), wu-w1))
        query = set(plant + random.sample(range(U), wq-w1))
        sm = Supermajority(U, wq/U, wu/U, w1/U, n, q_weight=1)
        sm.build([plant_point])
        res = any(sm.query(query))
        print(res)
        return res
    print(est(f, .1))

if __name__ == '__main__':
    main2()

