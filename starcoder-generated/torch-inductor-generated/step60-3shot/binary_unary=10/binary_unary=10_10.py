
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=8)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 + x2
        t3 = torch.relu(t2)
        return t3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, requires_grad=True)
x2 = torch.randn(2, 3, requires_grad=True)
y = m(x1, x2)

# Compute gradients
y.backward()









