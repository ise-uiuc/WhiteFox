
class Model(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_ff)
        self.activation = torch.nn.Linear(d_ff, d_model)
 
    def forward(self, x):
        v = self.linear(x)
        v = v.view(v.shape[0], 1, -1)
        v = self.activation(v)
        return v

# Initializing the model
d_model = 20
d_ff = 10
m = Model(d_model, d_ff)

# Inputs to the model
x = torch.randn(1, d_model)
