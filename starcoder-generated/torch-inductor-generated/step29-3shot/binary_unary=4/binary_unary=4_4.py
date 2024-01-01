
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1, other: Tensor):
        v1 = self.linear(x1)
        v2 = v1 + other
        return F.relu(v2)

# Initializing the model
module = Model()
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
