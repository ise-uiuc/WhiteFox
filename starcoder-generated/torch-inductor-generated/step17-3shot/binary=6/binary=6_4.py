
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
        self.linear.bias.data.copy_(torch.rand_like(self.linear.bias))
 
    def forward(self, x):
        v = self.linear(x)
        v_sub = v - self.linear.bias
        return v, v_sub

# Initializing the model
__hidden_units__ = 16
m = Model(32, hidden_units, True)

# Inputs to the model
x = torch.randn(1, 32)
