
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat([torch.nn.functional.relu(torch.nn.functional.pad(x1, (2, 3, 2, 3))), torch.nn.functional.relu(torch.nn.functional.pad(x1, (3, 3, 2, 3)))], 1)
        v2 = x1
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 10, 20)
