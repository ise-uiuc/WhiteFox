
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        
    def forward(self, x):
        y = self.linear1(x)
        z = torch.cat([y, y, y], 0)
        r = torch.tanh(z.view(-1, 4))
        return r

# Initializing the model
m = Model()


# Inputs to the model
x = torch.randn(3, 4)
