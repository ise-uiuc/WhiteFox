
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 327, bias=False)
        self.conv_transpose = torch.nn.ConvTranspose1d(327, 327, 3, padding=1, stride=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(10, 3)
