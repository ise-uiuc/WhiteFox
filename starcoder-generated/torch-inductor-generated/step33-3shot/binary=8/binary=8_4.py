
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8,  3, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x2
        return v3

# Initializing the model
m = Model()

# Set some random seed tensors as input to the model. We'll choose the same random seed tensors as those chosen in the previous example for convenience's sake.
torch.manual_seed(42)
x1 = torch.randn(1, 8, 64, 64)
torch.manual_seed(42)
x2 = torch.randn(1, 8, 64, 64)
