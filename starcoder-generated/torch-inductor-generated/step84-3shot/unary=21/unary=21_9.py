
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.max(torch.max(x, dim=-1, keepdim=False)[0], dim=-1, keepdim=False)[0]
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(256, 128, 75)
