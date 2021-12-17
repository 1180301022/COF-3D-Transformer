import torch
from model.tr_all import build_model
from QM7_CPE import QM7
from torch.utils.data import DataLoader

GPU = 1
model = build_model(vocab=23, tgt=1, k=23, dist_bar=[5, 10, 25]).cuda(GPU)
model.load_state_dict(torch.load('./params/epoch20.pkl'))
test_dataset = QM7(mode='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, drop_last=True)

criterion = torch.nn.MSELoss()
total_loss = 0
total_label = 0

# 测试
with torch.no_grad():
    for batch, (data, label) in enumerate(test_loader):
        atom, dist = data
        atom = atom.long().cuda(GPU)
        dist = dist.float().cuda(GPU)
        label = label.view(32, 1).cuda(GPU)
        mask = (atom != 0).unsqueeze(1).cuda(GPU)
        out = model(atom, mask, dist)
        loss = criterion(out, label)

        total_loss += abs(torch.sum(label, dim=0).item() - torch.sum(out, dim=0).item())
        total_label += abs(torch.sum(label, dim=0).item())

print(total_loss / total_label)
