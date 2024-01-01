
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight_init = torch.zeros(384, 256)
        self.linear = torch.nn.Linear(384, 256, bias=False)
        self.linear.weight.data = weight_init
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 384)
