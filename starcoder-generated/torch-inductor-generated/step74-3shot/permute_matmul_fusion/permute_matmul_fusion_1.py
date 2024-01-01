
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.bmm(v1, x2)
        return self.r(5.0 * (torch.bmm(v1, x2)) - 1.0)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
