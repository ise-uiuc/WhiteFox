
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1
        v2 = torch.sqrt(v1)
        v3 = torch.rsqrt(v2)
        v4 = torch.reciprocal(v3)
        v5 = torch.relu(v4)
        v6 = torch.tanh(v5)
        v7 = torch.sigmoid(v6)
        v8 = v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 512, 224, 224)
