
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 3, 1)
        v3 = v1.contiguous()
        return v1, v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64).float()
