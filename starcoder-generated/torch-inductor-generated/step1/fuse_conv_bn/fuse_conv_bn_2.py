 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, (1))
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x1):
        x1 = F.batch_norm(x1, 1, self.conv.bias, self.conv.weight, False, 1e-05, 0.9, False)
        