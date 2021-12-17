import torch
from model.tr_all import build_model
from QM7_CPE import QM7
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 超参数
batch_size = 32
learning_rate = 1e-4
epoch = 21
GPU = 1
max_atom_num = 23
target_size = 1
# log = SummaryWriter(log_dir='./log')

# 加载数据
train_dataset = QM7(mode='train')
# test_dataset = QM7(mode='test')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 模型
model = build_model(vocab=23, tgt=1, dist_bar=[5, 10, 25], k=23).cuda(GPU)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

step = 0

for temp_epoch in range(epoch):
    for batch, (data, label) in enumerate(train_loader):
        atom, dist = data
        atom = atom.long().cuda(GPU)
        dist = dist.float().cuda(GPU)
        label = label.view(batch_size, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, dist)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            print("Epoch:{}\tBatch:{}\tLoss:{}".format(temp_epoch, batch, loss))
            # log.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)
            step += 1

    # 保存模型
    if temp_epoch % 10 == 0:
        torch.save(model.state_dict(), './params/epoch{}.pkl'.format(temp_epoch))

# log.close()