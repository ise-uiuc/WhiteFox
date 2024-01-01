
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, min_value, max_value):
        v1 = x1.view(1, -1) # Reshape to a 1D tensor
        v2 = torch.clamp(v1, min_value, max_value) # Clamp to a minimum and maximum value
        return v2.view(3, 224, 224) # Reshape to have the same shape as the input tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 224, 224)
min_value, max_value = -1, 1
