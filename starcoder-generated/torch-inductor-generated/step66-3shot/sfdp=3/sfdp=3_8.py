
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads
        self.values_embedding = torch.nn.Embedding(embed_dim, embed_dim)
        self.keys_embedding = torch.nn.Embedding(embed_dim, embed_dim)
        self.queries_embedding = torch.nn.Embedding(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.scale_factor = torch.sqrt(torch.Tensor([self.head_dim])).to("cuda")
    
    def forward(self, x1, x2):
        qk = torch.matmul(self.queries_embedding(x1), self.keys_embedding(x1).transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(self.values_embedding(x2))
        return output

# Initializing the model
m = Model(embed_dim=embed_dim, num_heads=num_heads).to("cuda")

# Inputs to the model
x1 = torch.randint(embed_dim, (2,3)).to("cuda")
x2 = torch.randint(embed_dim, (4,5)).to("cuda")
