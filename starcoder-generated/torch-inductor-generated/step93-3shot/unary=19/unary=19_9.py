
class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.num_classed = num_classes
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(3, 6)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
