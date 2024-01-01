
class MyConvModule(nn.Conv2d):
    def __init__(self, cin, cout, kernel, stride=1, padding=0):
        super().__init__(cin, cout, kernel, stride=stride, padding=padding)
 
    def forward(self, x):
        return my_conv_fwd(self, x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.conv2 = MyConvModule(24, 64, 3, stride=2, padding=1)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x 
 
# Initialize model
model = MyModel()

# Inputs to the model
input = torch.randn(2, 3, 64, 64)
