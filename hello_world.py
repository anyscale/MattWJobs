import ray
import anyscale

@ray.remote
def say_hi(message):
    import time
    for i in range(200):
        print(i)
        time.sleep(5)
    return f"Hi hi, {message}."

ray.init()
print(ray.get(say_hi.remote("World")))