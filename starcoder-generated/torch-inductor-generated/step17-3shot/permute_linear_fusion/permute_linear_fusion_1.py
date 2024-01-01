
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.nn.functional.softmax(x1, dim=1)
# Inputs to the model
x1 = torch.randn(1, 10, 20, dtype=torch.float)
