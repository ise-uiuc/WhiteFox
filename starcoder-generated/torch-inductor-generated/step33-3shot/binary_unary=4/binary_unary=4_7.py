:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, input_tensor, __other__):
        v1 = self.linear(input_tensor)
        v2 = v1 + __other__
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5)
x2 = torch.randn(10)
