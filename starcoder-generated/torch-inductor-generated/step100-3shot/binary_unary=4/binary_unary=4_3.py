
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 12 * 12, 100)
 
    def forward(self, x1, input_tensor):
        v1 = self.linear(input_tensor)
        v2 = v1 + input_tensor
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 64 * 12 * 12)
