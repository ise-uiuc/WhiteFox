
class ReLUModule(torch.nn.Module):
    def __init__(self, max_value = 6.0):
        super().__init__()
        self.rnu=torch.nn.ReLU()
        self.max_value = max_value
    def forward(self, input):
        return torch.clamp(self.rnu(input), max=self.max_value)
# Inputs to the model
input = torch.randn(1, 2, 64, 64)
