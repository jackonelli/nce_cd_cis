import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_log():
    event_acc = EventAccumulator('res')
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags()['scalars'])

    # Plot loss
    step, loss, loss_p, loss_q = [], [], [], []
    for i, (l, lp, lq) in enumerate(zip(event_acc.Scalars('log-prob-aem-val'), event_acc.Scalars('log-prob-model-val'),
                              event_acc.Scalars('log-prob-proposal-val'))):

        step.append(l.step)
        loss.append(l.value)
        loss_p.append(lp.value)
        loss_q.append(lq.value)

    step = np.array(step)
    loss, loss_p, loss_q = np.array(loss), np.array(loss_p), np.array(loss_q)
    plt.plot(step, loss, label='loss, total')
    plt.plot(step, loss_p, label='loss, model')
    plt.plot(step, loss_q, label='loss, proposal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    read_log()