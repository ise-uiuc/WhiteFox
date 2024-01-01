
class Model(torch.nn.Module):
    def __init__(self, linear_out_features, linear_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(8, linear_out_features, bias=linear_bias)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model(32)
m_biased = Model(32, True)

# Inputs to the model
x1 = torch.randn(1, 8)
