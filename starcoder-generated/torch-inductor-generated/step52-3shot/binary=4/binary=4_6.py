
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.linear1 = torch.nn.Linear(16 * 3 * 3, 10)
 
    def forward(self, x):
        v1 = self.bn1(x)
        v2 = v1.contiguous().view(v1.shape[0], -1)
        