
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 9)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        sigmoid_result = torch.sigmoid(v1)
        v3 = v1 * sigmoid_result
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
