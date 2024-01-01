
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1
        v3 = v2 + x2
        return v3

# Initializing the model and setting the weights of the linear layer
m = Model()
m.linear.weight.data = torch.randn(32, 16)  # 32 is the output channel of the linear layer and 16 is the input channel of the linear layer
m.linear.bias.data = torch.randn(32)  # 32 is the output channel of the linear layer

# Input tensors
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 32)

# Check the calculation of the output tensor
