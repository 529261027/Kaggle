from tqdm.autonotebook import tqdm
import time
x = range(10000)
tk0 = tqdm(x, total=len(x))

for i in tk0:
    time.sleep(0.1)