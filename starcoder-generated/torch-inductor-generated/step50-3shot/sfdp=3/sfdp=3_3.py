
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model=7, num_heads=4)
 
    def forward(self, x1, x2):
        v1 = self.attention(x1, x2, x2)[0]
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5, 7)
x2 = torch.randn(4, 10, 7)
