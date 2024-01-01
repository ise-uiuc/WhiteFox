
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.proj_l = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout_p)
 
    def forward(self, x1, x2, x3):
        a1 = self.proj_l(x1)
        a2 = torch.zeros([16, 10, 256]).cuda()
        a3 = torch.randn([16, 5, 256]).cuda()
        