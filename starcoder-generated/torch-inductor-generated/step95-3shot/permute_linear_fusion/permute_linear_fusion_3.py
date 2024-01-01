
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.tanh(x1)
        v2 = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.relu(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
