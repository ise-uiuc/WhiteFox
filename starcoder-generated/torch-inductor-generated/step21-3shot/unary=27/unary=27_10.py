
class Model(torch.nn.Module):
    def __init__(self, in_features=20, n_class=10, min=0, max=0):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, n_class)
        self.m = torch.nn.ReLU6()
        self.fc2 = torch.nn.Linear(n_class, n_class, bias=True)
        self.min = min
        self.max = max

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.m(v1)
        v3 = self.fc2(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
