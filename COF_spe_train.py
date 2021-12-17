import torch
from tr_spe import build_model
from COF import COF
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 超参数
batch_size = 2
learning_rate = 1e-5
epoch = 23
GPU = 0
embed_dim = 516
max_atom_num = 2500
target_size = 1
encoder_layer_num = 6
head_num = 6
ffn_dim = 2048
log = SummaryWriter(log_dir='./log')

# 加载数据
train_dataset = COF(mode='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 网络初始化，训练设置
model = build_model(vocab=max_atom_num, tgt=1, embed_dim=embed_dim,
                    N=encoder_layer_num, head=head_num, ffn_dim=ffn_dim).cuda(GPU)
# model.load_state_dict(torch.load("./COFparams/SPE_epoch_10.pkl"))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# 训练
step = 0
for temp_epoch in range(epoch):
    for batch, (data, label) in enumerate(train_loader):
        atom, position = data
        atom = atom.cuda(GPU)
        position = position.float().cuda(GPU)
        label = label.float().view(batch_size, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, position)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示loss
        if batch % 20 == 0:
            print("Epoch:{}\tBatch:{}\tLoss:{}".format(temp_epoch, batch, loss))
            log.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)
            step += 1

    # 保存模型
    torch.save(model.state_dict(), './COFparams/SPE_area_epoch_{}.pkl'.format(temp_epoch))
log.close()
