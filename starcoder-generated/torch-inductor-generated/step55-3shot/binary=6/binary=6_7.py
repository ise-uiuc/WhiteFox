
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randint(0, 10, [5, 5])
        self.linear = torch.nn.Linear(10, 5, bias=False)
 
    def forward(self, x1):
        v4 = self.linear(x1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
other = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
