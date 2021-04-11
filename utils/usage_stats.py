import psutil
import pynvml
from pprint import pprint


def _get_cpu_usage():
    proc = psutil.Process()
    cpu_use = proc.cpu_percent()
    ram_percent = int(proc.memory_percent() * 100)
    ram_use = int(proc.memory_info().rss / 1024 / 1024)
    return cpu_use, ram_percent, ram_use
        