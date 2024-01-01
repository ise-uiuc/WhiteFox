
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            weight = torch.eye(3),
            bias = torch.ones(3),
        )
         
    def forward(self, x):
        v1 = F.linear(x.t(), self.weight, self.bias)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model. Bias is assigned to `1` because bias of `Linear` in PyTorch is initialized to `0` in the backend implementation
m = Model()

# Inputs to the model
x = torch.randn(20, 3)
