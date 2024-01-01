
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.quantized.FloatFunctional() # This is our custom API which supports the Quantization Aware Training
        self.linear = torch.nn.Linear(8, 8)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.input.linear(x1, self.linear.weight, self.linear.bias)        # v1 = linear(x1)
        v2 = self.input.sigmoid(v1)                                              # v2 = sigmoid(v1)
        v3 = v1 * v2                                                               # v3 = v1 * v2
        return v3, v2, v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
