
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=0)
    def forward(self, input): 
        return torch.tanh(self.conv2d(input))
# Inputs to the model
tensor = torch.randn(1, 3, 112, 112)
