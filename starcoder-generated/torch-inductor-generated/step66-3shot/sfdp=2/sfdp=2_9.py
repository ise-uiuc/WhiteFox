
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.inv_scale = 1.0 / (self.num_heads ** 0.5)
 
    def forward(self, query, value):
        scaled_qk = torch.matmul(query, value.transpose(-2, -1)).div(self.inv_scale)
        return scaled_qk

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
query = torch.randn(1, 16, 64, 64)
key = torch.randn(1, 32, 64, 64)
