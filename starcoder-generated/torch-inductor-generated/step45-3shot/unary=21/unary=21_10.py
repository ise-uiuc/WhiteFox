
class PatternModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        tensor = self.conv(x1)
        output_tensor = self.tanh(tensor)
        return (output_tensor)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
