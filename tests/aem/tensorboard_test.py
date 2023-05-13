import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def write():
    writer = SummaryWriter(logdir="res/")
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
    writer.close()

def load():
    event_acc = EventAccumulator('../../src/misc/res')
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags()['scalars'])

    # E.g. get wall clock, number of steps and value for a scalar 'y=2x'

    x, y = torch.zeros((100,)), torch.zeros((100,))
    for i, s in enumerate(event_acc.Scalars('y_2x')):
        x[i], y[i] = s.step, s.value

    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    load()
