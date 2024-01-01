
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = v1.reshape(-1, 2, 2)
        return v2
# Inputs to the model
x1 = torch.randn(4, 2, 2)
