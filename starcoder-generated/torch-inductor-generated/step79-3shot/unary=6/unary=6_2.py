
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.arange(0, 1 * 3 * 3 * 3, device=x1.device)
        v2 = v1.reshape(1, 3, 3, 3)
        v3 = v2.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        v4 = v3[0, 0]
        v5 = torch.arange(0, 1 * 3 * 3 * 3, device=x1.device)
        v6 = v5.reshape(1, 3, 3, 3)
        return v4 + v6[0][1, 0, 0]
# Inputs to the model
x1 = torch.randn(1, 3)
