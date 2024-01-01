
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3_t = torch.bmm(x2, v1) # v1.t()
        x3 = x3_t.permute(0, 2, 1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
