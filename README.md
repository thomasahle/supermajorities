# Supermajorities: An Evolutionary Data Structure for Sets
Code written for the blog post: https://thomasahle.com/blog/sets.html
and the article: https://arxiv.org/abs/1904.04045

Example of use:

```
    U = 10**3 + 9
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
    sm.build(data)
    
    print('Querying data structure')
    best = 0, None
    for point in sm.query(query):
        inter = len(point & query)
        if inter > best[0]:
            best = (inter, point)

    print(f'Best result was: {best}')
```
