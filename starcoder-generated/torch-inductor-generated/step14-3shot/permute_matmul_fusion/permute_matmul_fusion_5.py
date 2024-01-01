
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.unsqueeze(0)
        v1 = v1.permute(0, 2, 1)
        v2 = v1
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 2)
