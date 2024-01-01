
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.squeeze(x1, dim=-1)
        x3 = x2.transpose(1, 2)
        v3 = self.linear(x3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 1)
