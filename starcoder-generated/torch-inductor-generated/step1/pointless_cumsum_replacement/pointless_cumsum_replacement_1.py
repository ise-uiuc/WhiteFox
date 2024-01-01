
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        v1 = torch.full((64, 64), 1, dtype=torch.float32)
        v2 = torch.convert_element_type(v1, dtype=torch.float64)
        return torch.cumsum(v2, 1)


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

