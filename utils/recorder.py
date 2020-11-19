# @Time    : 2020/7/4
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : recoder.py
# @Project : utils/recoder.py
# @GitHub  : https://github.com/lartpang
import functools
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.misc import check_mkdir, construct_print


class TBRecorder(object):
    def __init__(self, tb_path):
        check_mkdir(tb_path)

        self.tb = SummaryWriter(tb_path)

    def record_curve(self, name, data, curr_iter):
        if not isinstance(data, (tuple, list)):
            self.tb.add_scalar(f"data/{name}", data, curr_iter)
        else:
            for idx, data_item in enumerate(data):
                self.tb.add_scalar(f"data/{name}_{idx}", data_item[name],
                                   curr_iter)

    def record_image(self, name, data, curr_iter):
        data_grid = make_grid(data, nrow=data.size(0), padding=5)
        self.tb.add_image(name, data_grid, curr_iter)

    def record_histogram(self, name, data, curr_iter):
        self.tb.add_histogram(name, data, curr_iter)

    def close_tb(self):
        self.tb.close()


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        construct_print(f"a new epoch start: {start_time}")
        func(*args, **kwargs)
        construct_print(
            f"the time of the epoch: {datetime.now() - start_time}")

    return wrapper


def CustomizedTimer(cus_msg):
    def Timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            construct_print(f"{cus_msg} start: {start_time}")
            results = func(*args, **kwargs)
            construct_print(
                f"the time of {cus_msg}: {datetime.now() - start_time}")
            return results

        return wrapper

    return Timer
