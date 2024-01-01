
class ModelTanh(torch.nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        v1 = x.permute(0,2,3,1)
        v1 = torch.tanh(v1)
        v2 = v1.permute(0,3,1,2)
        return v2
# Inputs to the model
x = torch.randn(1, 224, 224, 3)
