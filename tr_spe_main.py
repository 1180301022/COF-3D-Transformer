import torch
from model.tr_spe import build_model
from QM7_APE import QM7
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 超参数
batch_size = 32
learning_rate = 1e-5
epoch = 11
GPU = 1
max_atom_num = 23
target_size = 1
# log = SummaryWriter(log_dir='./log')

# 加载数据
train_dataset = QM7(mode='train')
test_dataset = QM7(mode='test')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 网络初始化，训练设置
model = build_model(max_atom_num, target_size).cuda(GPU)  # 5种原子，1个输出值
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# 训练
step = 0
for temp_epoch in range(epoch):
    for batch, (data, label) in enumerate(train_loader):
        atom, position = data
        atom = atom.long().cuda(GPU)
        position = position.cuda(GPU)
        label = label.view(batch_size, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, position)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示loss
        if batch % 20 == 0:
            print("Epoch:{}\tBatch:{}\tLoss:{}".format(temp_epoch, batch, loss))
            # log.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)
            # step += 1

    # 保存模型
    # if temp_epoch % 10 == 0:
    #     torch.save(model.state_dict(), './epoch{}.pkl'.format(temp_epoch))
# log.close()

# 测试
with torch.no_grad():
    for batch, (data, label) in enumerate(test_loader):
        atom, position = data
        atom = atom.long().cuda(GPU)
        position = position.cuda(GPU)
        label = label.view(32, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, position)
        loss = criterion(out, label)
        print("Batch:{}\tLoss:{}".format(batch, loss))
