
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x.unsqueeze(0)
        v2 = torch.tanh(v1)
        v3 = v2.squeeze()
        return v3
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
