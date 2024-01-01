
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(num_head, num_head)
 
    def forward(self, x1):
        v1, _ = self.attention(x1, x1, x1, attn_mask=x1)
        return v1

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1, 128, 128)
__attn_mask__ = x1
