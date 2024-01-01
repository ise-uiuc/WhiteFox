
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.attention = torch.nn.MultiheadAttention(d_model=512, num_heads=8)
 
    def forward(self, x1, x2):
        v1, v2 = self.attention(query=x1, key=x2, value=x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 512)
x2 = torch.randn(1, 3, 512)
