
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 8)
        self.add = torch.nn.quantized.FloatFunctional()
 
    def forward(self, x1, x2):
        t1 = self.l1(x1)
        t2 = self.add.add_scalar(t1, 1)
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
