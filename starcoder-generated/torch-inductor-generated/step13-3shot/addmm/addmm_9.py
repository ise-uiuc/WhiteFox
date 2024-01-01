
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10).to(torch.float32)
        self.conv = torch.nn.Conv2d(
                10, 10, 3
            ).to(torch.float32)
        self.dropout = torch.nn.Dropout()
    def forward(self, inp, x1, x2):
        v1 = self.linear(inp)
        v2 = self.conv(x1)
        v3 = self.dropout(v2)
        v3 = v3 + x2
        return v3
# Inputs to the model
inp = torch.randn(1, 10, requires_grad=True)
x1 = torch.randn(1, 10, 10, 10)
x2 = torch.randn(1, 10, 10, 10)
