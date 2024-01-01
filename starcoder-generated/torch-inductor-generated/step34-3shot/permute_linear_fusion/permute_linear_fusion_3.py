
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = torch.nn.Squeeze(-1)

    def forward(self, x1, x2, x3):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, x2, x3)
        v3 = self.squeeze(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(2, 6)
x3 = torch.randn(6)
