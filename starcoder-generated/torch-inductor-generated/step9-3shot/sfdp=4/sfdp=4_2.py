
class Model(torch.nn.Module):
    def __init__(self, n_head, n_hidden_attn):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.n_hidden_attn = n_hidden_attn
        self.dropout_attn = torch.nn.Dropout(p=0.6)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.n_head_dim = self.n_hidden_attn // self.n_head
        self.n_all_head_dim = self.n_head * self.n_head_dim
        self.key = torch.nn.Linear(self.n_emb, self.n_all_head_dim)
        self.value = torch.nn.Linear(self.n_emb, self.n_all_head_dim)
        self.query = torch.nn.Linear(self.n_emb, self.n_all_head_dim)
        self.layernorm = torch.nn.LayerNorm(self.n_emb)
 
    def forward(self, x1):
        x2 = self.key(x1).reshape(x1.size(0), x1.size(1), self.n_head, self.n_head_dim).transpose(1,2)
        x3 = self.value(x1).reshape(x1.size(0), x1.size(1), self.n_head, self.n_head_dim).transpose(1,2)
        x4 = self.query(x1).reshape(x1.size(0), x1.size(1), self.n_head, self.n_head_dim).transpose(1,2)
        v1 = (x4 @ x3.transpose(-2, -1)) / math.sqrt(self.n_head_dim)
        v1 = v1 + getattr(self, 'attn_mask', _create_attn_mask(x1, x1, x1))
        v2 = self.dropout_attn(self.softmax(v1))
        v3 = (v2 @ x3).reshape(v2.size(0), v2.size(1), -1)
        v4 = self.dropout(v3)
        v5 = self.layernorm(x1 + v4)
        return v5

# Initializing the model
m = Model(n_head=8, n_hidden_attn=512)
# Inputs to the model
x1 = torch.randn(28, 64, 512)
