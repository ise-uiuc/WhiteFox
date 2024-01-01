
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.squeeze(dim=1)
        v2 = torch.tanh(v1)
        return v2.unsqueeze(dim=1)
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
