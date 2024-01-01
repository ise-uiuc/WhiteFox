
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=128, out_features=256, bias=False)
        self.linear.weight.data = torch.eye(256, 128) * 15 + torch.rand(256, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
