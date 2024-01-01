
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.relu(torch.sum(x1, dim=[2, 3]) + torch.sum(x1, dim=[2, 3]) + torch.sum(x1, dim=[2, 3]))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
