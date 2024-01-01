
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shared_weights = torch.nn.parameter.Parameter(torch.Tensor(1, 1, 64, 8)) # Note that the shapes of the shared weights are (1, 1, 64, 8). This means that the shared weights can contain one example, one batch, 64 values for each input, and 8 shared weights. This might be a typical pattern found in Transformers when they are initialized like this.

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, self.shared_weights)
        v2 = torch.matmul(x2, self.shared_weights)
        v3 = torch.matmul(x3, self.shared_weights)
        
        return v1, v2, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)
