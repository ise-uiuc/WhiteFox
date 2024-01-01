
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.sigmoid(x1)
        x3 = x1 * x2
        return x3

# Initializing the model
model = Model()

# Inputs to the model
input = torch.randn(1, 512)
