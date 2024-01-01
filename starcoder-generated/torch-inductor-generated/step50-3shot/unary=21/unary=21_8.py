
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 512, bias=False)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(2, 128)
