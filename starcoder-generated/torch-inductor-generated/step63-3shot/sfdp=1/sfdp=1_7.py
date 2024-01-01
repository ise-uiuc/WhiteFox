
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
 
    def forward(self, query, key, value, mask, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
embed_dim = 4
num_heads = 2
m = Model(embed_dim, num_heads)

# Inputs to the model
query = torch.randn(3, 3, embed_dim)
key = torch.randn(4, 3, embed_dim)
value = torch.randn(4, 3, embed_dim)
mask = torch.randn(3, 4)
inv_scale_factor = torch.randn(num_heads, 1) # This is randomly initialized to prevent scaling the attention probabilities
dropout_p = 0.2
