
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()

    def forward(input_tensor):
        t1 = torch.tanh(input_tensor)
        t2 = torch.nn.functional.gelu(t1)
        t3 = torch.clamp_min(t2, min_value)
        k6 = torch.clamp_max(t3, max_value)
        return torch.sigmoid(k6)

# Initializing the model
m = Model(min_value=-0.5, max_value=0.5)

# Inputs to the model
x = torch.randn(2, 2, 2)
