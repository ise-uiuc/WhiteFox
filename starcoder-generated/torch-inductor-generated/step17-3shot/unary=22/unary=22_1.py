
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8 * 5 * 5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(-1, 8 * 5 * 5))
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8 * 5 * 5)