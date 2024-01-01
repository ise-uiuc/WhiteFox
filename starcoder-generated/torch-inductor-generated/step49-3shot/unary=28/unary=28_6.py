
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=5, out_features=8)
 
    def forward(self, input_1, v1, v2):
        v3 = torch.matmul(input_1, self.linear.weight) + self.linear.bias
        v4 = torch.clamp(v3, min=v1, max=v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5)
w1 = float('-inf')
w2 = float('inf')
