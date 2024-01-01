
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Linear(10, 5)
        self.op2 = torch.nn.Softmax(dim=1)
        self.op3 = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.op1(x1)
        y1 = self.op2(v1)
        y2 = self.op3(v1)
        return y1, y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
