
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.nn.functional.relu(torch.add(x1, 10))
# Inputs to the model
x1 = torch.randn(3, 4)
