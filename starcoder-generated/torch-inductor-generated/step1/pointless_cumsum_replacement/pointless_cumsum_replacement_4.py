
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.full((3, x.size(2)), 1, dtype=torch.float32, device=x.device)
        v2 = torch.convert_element_type(v1, dtype=torch.float32)
        v3 = torch.cumsum(v2, 1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

# Outputs of the model
