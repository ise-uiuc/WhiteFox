
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create two conv2d layers with 3 input channels and 4 output channels
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    # Add a sigmoid activation function to the output of the second convolution to make two sigmoid activations in a row
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = torch.sigmoid(t1)
        t3 = self.conv2(t2)
        t4 = torch.sigmoid(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

