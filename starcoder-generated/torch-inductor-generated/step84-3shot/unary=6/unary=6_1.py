
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        t1 = self.relu(x1)
        t2 = x1 * t1
        t3 = x1 / t2
        return t3.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
