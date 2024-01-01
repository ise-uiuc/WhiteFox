
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = v2 + additional_tensor
        return v3

# Initializing `other`
other = torch.randn(1, 1, 64, 64)
# Initializing `additional_tensor` to pass it as a keyword argument to the model
additional_tensor = torch.randn(1, 1, 64, 64)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
