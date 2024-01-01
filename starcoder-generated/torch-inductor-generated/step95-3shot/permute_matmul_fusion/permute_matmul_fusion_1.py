
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.type(torch.int)
        x2 = x2.type(torch.int)
        x1_permute = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(x1_permute, x2)
        v2 = torch.add(v3, v3)
        v1 = torch.reshape(v2, (1, -1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
