
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = v1 + 3
        v3 = nn.functional.leaky_relu(v2, 0.1)
        v4 = nn.functional.hardtanh_(v3, min_val=0., max_val=6.)
        v5 = nn.functional.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
