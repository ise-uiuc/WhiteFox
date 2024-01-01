
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model
m = Model()
torch.nn.init.normal_(m.linear.weight, std=0.01)
torch.nn.init.normal_(m.linear.bias, std=0.01)

# Inputs to the model, "other" should have the same shape as output(m(x1))
x1 = torch.randn(1, 5)
other = torch.randn(1, 3)
