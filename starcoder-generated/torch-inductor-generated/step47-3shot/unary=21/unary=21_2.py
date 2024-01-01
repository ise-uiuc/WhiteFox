
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features=32, out_features=64, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, z):
        w1 = self.conv(z)
        v3 = self.tanh(w1)
        g1 = self.linear(v3)
        u1 = self.relu(g1)
        v5 = self.tanh(u1)
        return v5
# Inputs to the model
u = torch.randn(1, 3, 4, 4, requires_grad=True)
