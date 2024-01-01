
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 - self.x2
        v2 = F.relu(v1)
        v3 = v2 - self.x3
        v4 = F.relu(v3)
        v5 = v4 - self.x1
        v6 = F.relu(v5)
        v7 = v6 - self.x1
        v8 = F.relu(v7)
        return v8

