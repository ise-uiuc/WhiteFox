
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - other
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 128, 128)
