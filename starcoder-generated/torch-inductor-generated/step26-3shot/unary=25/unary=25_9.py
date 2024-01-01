
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        tmp0 = 0.1
        v3 = tmp0 > 0
        v4 = v1[v3]
        v5 = v1 * tmp0
        v6 = torch.where(v3, v1, v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(123, 10)
