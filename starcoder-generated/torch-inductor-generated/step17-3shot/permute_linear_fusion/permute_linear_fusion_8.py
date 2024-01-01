
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.tanh(v2)
        v3 = v3 + v2
        v4 = x1.permute(0, 2, 1)
        v3 = v3 + v4
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
