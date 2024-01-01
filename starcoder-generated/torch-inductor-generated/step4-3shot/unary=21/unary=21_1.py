
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels=3, out_channels=257, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3)
