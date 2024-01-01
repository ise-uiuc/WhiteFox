
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, min_value=0, max_value=1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Getting the weights of the model
w_linear = m.linear.weight.data.clone().detach()
b_linear = m.linear.bias.data.clone().detach() if m.linear.bias is not None else None

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
