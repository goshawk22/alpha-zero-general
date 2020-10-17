import ray
ray.init()

y = 12
ref = ray.put(y)


@ray.remote
def calculate(val):
    y = ray.get(ref)
    return y*val


futures = [calculate.remote(i) for i in range(4)]
print(ray.get(futures))
