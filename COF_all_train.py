import torch
from tr_all import build_model
from COF import COFWithDistanceMatrix
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 超参数
batch_size = 2
learning_rate = 1e-5
epoch = 23
GPU = 2
embed_dim = 516
max_atom_num = 2500
target_size = 1
encoder_layer_num = 6
head_num = 6
ffn_dim = 2048
dist_bar = [5, 10, 25, 100]
sample_num = 1000
log = SummaryWriter(log_dir='./log/all')

# 加载数据
train_dataset = COFWithDistanceMatrix()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 模型
model = build_model(vocab=max_atom_num, tgt=target_size, dist_bar=dist_bar, k=sample_num,
                    ffn_dim=ffn_dim, embed_dim=embed_dim, head=head_num).cuda(GPU)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

step = 0

for temp_epoch in range(epoch):
    for batch, (data, label) in enumerate(train_loader):
        atom, _, matrix = data
        atom = atom.cuda(GPU)
        # pos3d = pos3d.cuda(GPU)
        matrix = matrix.float().cuda(GPU)
        label = label.view(batch_size, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, matrix)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            print("Epoch:{}\tBatch:{}\tLoss:{}".format(temp_epoch, batch, loss))
            log.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)
            step += 1

    torch.save(model.state_dict(), './COFparams/COF_all_epoch_{}.pkl'.format(temp_epoch))

log.close()