
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.rand(1, 3, 64, 64).fill_(2)
        return v1
# Inputs to the model
x1 = torch.zeros(1, 3, 64, 64)
