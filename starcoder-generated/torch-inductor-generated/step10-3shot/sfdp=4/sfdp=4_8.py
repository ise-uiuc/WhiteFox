
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(17, 2)
 
    def forward(self, q, k, v):
        a, _ = self.multihead_attention(q, k, v)
        return a

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 10, 17)
k = torch.randn(1, 20, 17)
v = torch.randn(1, 20, 2)
