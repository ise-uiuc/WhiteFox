 2 
class Model():
    def __init__(self):
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x2):
        w1 = self.conv(x2)
        w2 = torch.sigmoid(w1)
        w3 = w1 * w2
        return w3

# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
