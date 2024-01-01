
class Model(torch.nn.Module):
    def __init__(self, min, max,):
        super().__init__()
 
    def forward(self, input_tensor,):
        t1 = linear(input_tensor)
        t2 = torch.clamp_min(t1, min_value)
        t3 = torch.clamp_max(t2, max_value)
        return t3

# Initializing the model
m = Model(min, max)

# Inputs to the model
input_tensor = torch.randn(1,3,224,224)
