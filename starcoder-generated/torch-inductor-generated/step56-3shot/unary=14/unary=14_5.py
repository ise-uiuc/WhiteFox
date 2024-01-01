
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], 1)
        v2 = torch.unsqueeze(v1, 2)
        v3 = torch.squeeze(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 3, 4)
x2 = torch.randn(1, 4, 3, 4)
