
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output.reshape(-1, self.embed_dim, self.num_heads)

# Initializing the model
embed_dim = 512
num_heads = 8
m = Model(embed_dim, num_heads)

# Inputs to the model
query = torch.randn(1, 512, 64, 64)
key = torch.randn(1, 512, 64, 64)
value = torch.randn(1, 512, 64, 64)
scale_factor = torch.tensor([embed_dim ** -0.5])
dropout_p = 0.2
