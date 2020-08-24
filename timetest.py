from utils.load_data import get_data
import time

data = get_data()

start = time.time()

for epoch in range(5):
    for batch in data.trainloader:
        pass

end = time.time()

print(end - start)

