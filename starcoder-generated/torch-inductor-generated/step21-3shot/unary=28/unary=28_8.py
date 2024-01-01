
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu6(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
_ = torch.manual_seed(42)
x1 = torch.randn(1, 64)
