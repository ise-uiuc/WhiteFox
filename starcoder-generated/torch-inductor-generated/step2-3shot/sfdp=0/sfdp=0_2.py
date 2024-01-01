
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
 
    def init_multihead(self):
        self.heads = nn.ModuleList([nn.MultiheadAttention(self.hidden_dim, self.n_heads) for _ in range(self.n_layers)])
 
    def forward(self, x1, x2):
        for layer in self.heads:
            x1, x2 = layer(x1, x2)
        return x1, x2

# Inputs to the model
hidden_dim = 512
n_heads = 8
n_layers = 6
attn = Attention(hidden_dim, n_heads, n_layers)
attn.init_multihead()
x1 = torch.rand(1, 100, hidden_dim)
x2 = torch.rand(1, 100, hidden_dim)
output1, output2 = attn(x1, x2)
output3, output4 = attn(output1, output2)
