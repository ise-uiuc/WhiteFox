
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #...
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=3)
        
        while random.choice([0, 1]):
            #...
        
    def forward(self, x1):
        #...
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)

        if random.choice([0, 1]):
            v3 = self.conv1(x1)
            v4 = torch.sigmoid(v3)
        else:
            v3 = v4 = torch.randn(v1)

        return v3, v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
