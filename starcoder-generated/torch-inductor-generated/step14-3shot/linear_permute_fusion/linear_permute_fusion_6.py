
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1)
        v2 = v1.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
