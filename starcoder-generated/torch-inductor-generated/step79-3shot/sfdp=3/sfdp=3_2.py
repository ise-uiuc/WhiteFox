
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention()
 
    def forward(self, q, k, v):
        output, output_weights = self.attention(q, k, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 3, 1)
k = torch.randn(1, 8, 1, 5)
v = torch.randn(1, 8, 1, 5)

