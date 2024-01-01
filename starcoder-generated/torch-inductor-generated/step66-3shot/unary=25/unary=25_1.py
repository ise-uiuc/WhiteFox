
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v1_t = v1.transpose(1, 0)  # Transpose the input tensor for easier boolean indexing
        negative_slope = 0.1
        v2 = v1_t > 0
        v3 = v1_t * negative_slope
        v4 = torch.where(v2, v1_t, v3)
        return v4.transpose(1, 0)

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(64, 64)
