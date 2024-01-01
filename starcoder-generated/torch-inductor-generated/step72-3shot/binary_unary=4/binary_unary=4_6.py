
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64*64*3, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 32)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear1(x3)
        v2 = self.linear2(v1)
        v3 = self.linear3(v2)
        v4 = v1 + x1
        v5 = relu(v4)
        v6 = v3 + x2
        v7 = relu(v6)
        v8 = v5 + v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64*64*3)
y1 = torch.randn(1, 32)
x2 = torch.randn(1, 32)
x3 = torch.randn(1, 64*64*3)
