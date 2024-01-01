
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, torch.randn(2, 2), torch.randn(2))
        a1 = v1.numpy()
        v2 = a1.permute((0, 2, 1))
        a2 = torch.from_numpy(v2).float()
        return a2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
