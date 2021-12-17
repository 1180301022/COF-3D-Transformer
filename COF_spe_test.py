import torch
from tr_spe import build_model
from COF import COF
from torch.utils.data import DataLoader

# 超参数
batch_size = 2
GPU = 5
embed_dim = 516
max_atom_num = 2500
target_size = 1
encoder_layer_num = 6
head_num = 6
ffn_dim = 2048

# 加载数据
test_dataset = COF(mode='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 网络初始化，训练设置
model = build_model(vocab=max_atom_num, tgt=1, embed_dim=embed_dim,
                    N=encoder_layer_num, head=head_num, ffn_dim=ffn_dim).cuda(GPU)
model.load_state_dict(torch.load("./COFparams/SPE_area_epoch_13.pkl"))

total_loss = 0
total_label = 0

with torch.no_grad():
    for data, label in test_loader:
        atom, position = data
        atom = atom.long().cuda(GPU)
        position = position.cuda(GPU)
        label = label.view(batch_size, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, position)

        total_loss += abs(torch.sum(label, dim=0).item() - torch.sum(out, dim=0).item())
        total_label += abs(torch.sum(label, dim=0).item())
        print(total_loss / total_label)
