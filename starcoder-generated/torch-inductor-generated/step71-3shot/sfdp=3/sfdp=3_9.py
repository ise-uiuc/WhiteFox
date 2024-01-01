
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.Linear(10, 10)
 
    def forward(self, x):
        weight = self.matmul(x)
        return weight.softmax(dim=-1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
