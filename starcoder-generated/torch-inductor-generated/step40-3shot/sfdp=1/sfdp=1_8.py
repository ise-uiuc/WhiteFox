
class Model(torch.nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.multi_head = torch.nn.MultiheadAttention(embed_dim, num_heads)
 
    def forward(self, x):
        output, _ = self.multi_head(x, x, x)
        return output

# Initializing the model
m = Model(5, 30)

# Input to the model
x = torch.randn(1, 5, 30, 40)
