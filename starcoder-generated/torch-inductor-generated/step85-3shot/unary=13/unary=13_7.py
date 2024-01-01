
class Model(torch.nn.Module):
    def __init__(self, input_tensor_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_tensor_size, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
input_tensor_size = 8
m = Model(input_tensor_size)

# Inputs to the model
x1 = torch.randn(1, input_tensor_size)
