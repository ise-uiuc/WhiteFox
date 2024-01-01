
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32, bias=True)
        self.t1 = torch.nn.ConvTranspose2d(1, 2, 1, 2)
        self.l2 = torch.nn.Linear(32, 15, bias=True)
    def forward(self, x1):
        t1 = self.l1(x1)
        v1 = torch.relu(t1)
        t2 = self.t1(v1)
        v2 = torch.relu(t2)
        t3 = self.l2(v2)
        v3 = torch.relu(t3)
        v4 = torch.sigmoid(t3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 1)
