import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    :param global_step:记录当前执行步数
    :param learning_rate_base: 预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降
    :param total_steps: 总训练步数，=epoch * sample_count / batch_size
    :param warmup_learning_rate: warm up阶段线性增长初始值
    :param warmup_steps: warm up需要持续的步数
    :param hold_base_rate_steps:可选参数，当warm up阶段结束后保持学习率不变，直到hold_base_rate_steps结束后才开始学习率下降
    :return:
    """
    if total_steps < warmup_steps:
        raise ValueError("total_steps must be larger or equal to warmup_steps.")

    # 这里实现了余弦退火原理，设置学习率最小为0，所以简化表达式如下
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
                                                           float(total_steps - warmup_steps - hold_base_rate_steps)))

    # 如果hold_base_rate_steps大于0，表面在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)

    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError("learning_rate_base must be larger or equal to warmup_learning_rate.")

        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        # 只有当global_step仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)

    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose

        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.learning_rates = []

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print("\nBatch %05d: setting learning rate to %s." % (self.global_step + 1, lr))


# pytorch的实现过程
if __name__ == '__main__':
    pass


# # tensorflow的实现过程
# if __name__ == '__main__':
#     # Create a model.
#     model = Sequential()
#     model.add(Dense(32, activation='relu', input_dim=100))
#     model.add(Dense(10, activation='softmax'))
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     # 样本总数
#     sample_count = 12608
#     # Total epochs to train.
#     epochs = 50
#     # Number of warmup epochs.
#     warmup_epoch = 10
#     # Training batch size, set small value here for demonstration purpose.
#     batch_size = 16
#     # Base learning rate after warmup.
#     learning_rate_base = 0.0001
#
#     total_steps = int(epochs * sample_count / batch_size)
#
#     # Compute the number of warmup batches.
#     warmup_steps = int(warmup_epoch * sample_count / batch_size)
#
#     # Generate dummy data.
#     data = np.random.random((sample_count, 100))
#     labels = np.random.randint(10, size=(sample_count, 1))
#
#     # Convert labels to categorical one-hot encoding.
#     one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
#
#     # Compute the number of warmup batches.
#     warmup_batches = warmup_epoch * sample_count / batch_size
#
#     # Create the Learning rate scheduler.
#     warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
#                                             total_steps=total_steps,
#                                             warmup_learning_rate=4e-06,
#                                             warmup_steps=warmup_steps,
#                                             hold_base_rate_steps=5,
#                                             )
#
#     # Train the model, iterating on the data in batches of 32 samples
#     model.fit(data, one_hot_labels, epochs=epochs, batch_size=batch_size,
#               verbose=0, callbacks=[warm_up_lr])
#
#     import matplotlib.pyplot as plt
#
#     plt.plot(warm_up_lr.learning_rates)
#     plt.xlabel('Step', fontsize=20)
#     plt.ylabel('lr', fontsize=20)
#     plt.axis([0, total_steps, 0, learning_rate_base * 1.1])
#     plt.xticks(np.arange(0, epochs, 1))
#     plt.grid()
#     plt.title('Cosine decay with warmup', fontsize=20)
#     plt.show()
