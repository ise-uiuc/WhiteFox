
class Model(torch.nn.Module):
    def __init__(self, w11, w12, w13):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.linear.weight.data = torch.Tensor([w11, w12, w13])
        self.linear.bias.data = torch.Tensor([1.0, 1.0, 1.0])
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1

# Initializing the model
m = Model(0, 2, 4)

# Inputs to the model
x1 = torch.randn(1, 3)
