
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
        self.fc = torch.nn.Linear(5632, 16)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = F.max_pool2d(v1, kernel_size=2)
 
        v2 = v1.view(-1, 5632)
        v3 = self.fc(v2)
 
        v4 = v3 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
