
class Model(torch.nn.Module):
    # Declare this model has one input and one output
    def __init__(self):
        super().__init__()
        self.input = torch.randn(1, 1, 1, 1, requires_grad=True) # Initialize the input to the model
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 1, 1, stride=1, padding=0)
    def forward(self):
        v1 = self.conv1(self.input)
        v2 = self.conv2(v1)
        return v2

# Inputs to the model
x1 = torch.randn(3, 16, 28, 28)
x2 = torch.randn(3, 1, 28, 28)
