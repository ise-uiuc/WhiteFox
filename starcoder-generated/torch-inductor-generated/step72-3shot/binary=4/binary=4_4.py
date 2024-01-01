
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=1,padding=1)
        self.pooling = torch.nn.AvgPool2d(2, stride=2, padding=0)
 
    def forward(self, input):
        v1 = self.pooling(self.conv(input))
        v2 = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])
        return v1 + v2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3, 64, 64)
