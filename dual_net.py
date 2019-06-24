import glob
import random

import numpy as np

import face
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, prefix, shuffle=True):
        self.file_list = glob.glob(prefix + '*')
        self.data = []
        self.data_num = 0
        self.idx = 0

        self.load(shuffle)

    def parse_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        assert(len(lines) % 3 == 0)
        for i in range(0, len(lines), 3):
            fc = face.FaceCube()
            fc.from_string(lines[i][:-1])
            cc = fc.to_cubie_cube()
            inp = cc.to_one_hot().flatten()
            po = np.zeros(18)
            po[int(lines[i + 1])] = 1
            # po = int(lines[i + 1])
            val = 1 / float(lines[i + 2])

            self.data.append([inp, po, val])

    def load(self, shuffle):
        if len(self.file_list) == 0:
            print('No data to load!')

        for f in self.file_list:
            self.parse_file(f)

        self.data_num = len(self.data)

        print('Load {} training samples from {} file(s).'.format(
            self.data_num, len(self.file_list)))

        if shuffle:
            print('Shuffling trianing data ...')
            random.shuffle(self.data)

    def sample(self, batch_size=256):
        assert self.data_num > batch_size

        if self.idx + batch_size >= self.data_num:
            self.idx = 0

        batch = self.data[self.idx: self.idx + batch_size]
        inp = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        # po = torch.tensor([b[1] for b in batch], dtype=torch.long)
        # po = torch.squeeze(po)
        po = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        val = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1)

        self.idx += batch_size
        return inp, po, val


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(480, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.l2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)

        # policy head
        self.p1 = nn.Linear(2048, 512)
        self.bnp1 = nn.BatchNorm1d(512)
        self.p2 = nn.Linear(512, 18)

        # value head
        self.v1 = nn.Linear(2048, 512)
        self.bnv1 = nn.BatchNorm1d(512)
        self.v2 = nn.Linear(512, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = F.elu(out)

        out_p = self.p1(out)
        out_p = self.bnp1(out_p)
        out_p = self.p2(out_p)

        out_v = self.v1(out)
        out_v = self.bnv1(out_v)
        out_v = self.v2(out_v)
        out_v = torch.sigmoid(out_v)

        return out_p, out_v


train_dl = DataLoader('train/train')
test_dl = DataLoader('train/test')
model = Model()

p_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
# p_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
v_loss_fn = torch.nn.MSELoss(reduction='mean')

learning_rate = 0.01
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

train_p_loss = []
test_p_loss = []
train_v_loss = []
test_v_loss = []

for i in range(2000):
    # scheduler.step()

    x, p, v = train_dl.sample()
    p_, v_ = model.forward(x)

    p_loss = p_loss_fn(p_, p)
    v_loss = v_loss_fn(v_, v)
    loss = p_loss + v_loss

    train_p_loss.append(p_loss)
    train_v_loss.append(v_loss)

    print('Iter {}, P loss {:.4f}, V loss {:.4f}, Total loss {:.4f}'.format(
        i, p_loss.item(), v_loss.item(), loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        x, p, v = test_dl.sample()
        p_, v_ = model.forward(x)

        p_loss = p_loss_fn(p_, p)
        v_loss = v_loss_fn(v_, v)
        loss = p_loss + v_loss

        test_p_loss.append(p_loss)
        test_v_loss.append(v_loss)

        mp = torch.argmax(p, 1)
        mp_ = torch.argmax(p_, 1)
        print(mp)
        print(mp_)
        print(float((mp == mp_).sum()) / x.shape[0])

        print('Test iter {}, P loss {:.4f}, V loss {:.4f}, Total loss {:.4f}'.format(
            i, p_loss.item(), v_loss.item(), loss.item()))


plt.plot(train_p_loss, label='policy loss (train)')
plt.plot(test_p_loss, label='policy loss (test)')
plt.plot(train_v_loss, label='value loss (train)')
plt.plot(test_v_loss, label='value loss (test)')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.show()
