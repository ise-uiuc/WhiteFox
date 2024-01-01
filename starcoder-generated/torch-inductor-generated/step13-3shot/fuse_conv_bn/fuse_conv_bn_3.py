
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2) 
        self.conv2 = torch.nn.Conv2d(1, 1, 2) 
        self.drop2d = torch.nn.Dropout2d()
        self.relu = torch.nn.ReLU() 
    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.drop2d(x)
        x = self.relu(x)

        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
