
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = torch.nn.functional.relu(v2)
        # No t7, t8, t9 in the model, but the following
        # v7 = torch.nn.functional.leaky_relu(v3)
        v8 = torch.nn.functional.relu(v3, negative_slope=1)
        #v10 = torch.nn.functional.selu(v3, alpha=1)
        return v3, v8#, v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.tensor([2.0])
x3 = torch.randn(1, 3)
y1 = m(x1, x2, x3)

