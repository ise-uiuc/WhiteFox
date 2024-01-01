
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, y):
        v1 = self.conv(x)
        y_len = y.size(0)
        v2 = y.view(1, y_len)
        v3 = torch.cat([v1, v2], 1)
        v4 = torch.full((y_len,), -1, dtype=torch.int32)
        v5 = v3[:, v4]
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
y = torch.randn(10) # Assume the size of input y should be between [0, 17179869184]
