
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1_t = self.linear(x1)
        v1 = sigmoid(v1_t)
        v3 = v1 * v1_t
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
