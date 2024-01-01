
class Model(torch.nn.Module):
    def __init__(self, num_heads: int = 2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
 
    def forward(self, x1):
        # x1 is the query
        # x2 should be padding
        x2 = torch.zeros_like(x1)
        x3 = torch.zeros_like(x1)
        output, attn = self.multihead_attn(query=x1, key=x1, value=x1, key_padding_mask=x2, need_weights=x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model and their shape (please change the shape according to your model's inputs)
x1 = torch.randn(1, 20, 512)
