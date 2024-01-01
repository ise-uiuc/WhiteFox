
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat(tensors=(x1, x1), dim=1)
        v2 = torch.cat(tensors=(v1, v1), dim=1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2)
