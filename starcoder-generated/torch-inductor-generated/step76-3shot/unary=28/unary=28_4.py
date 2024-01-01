
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16,bias=True)
 
    def forward(self, input_tensor, min_value=0.0, max_value=1.0):
        v = self.linear(input_tensor)
        if min_value is not None:
            v = torch.clamp_min(v, min_value)
        if max_value is not None:
            v = torch.clamp_max(v, max_value)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
