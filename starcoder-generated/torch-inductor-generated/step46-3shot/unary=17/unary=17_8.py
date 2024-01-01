
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.abs(x)
        v2 = torch.relu(x)
        v3 = v1+v2
        return v3
# Inputs to the model
x = torch.randn(1,3,8,8)
