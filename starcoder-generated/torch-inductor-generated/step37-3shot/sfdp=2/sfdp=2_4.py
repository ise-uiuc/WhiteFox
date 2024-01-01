
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 32
        self.head = 8
        self.num_patch = int(8 * 8 + 4 * 4 + 2 * 2 + 1)
        self.pos_dim = self.emb_dim // self.head
        self.query_emb = torch.nn.Embedding(self.num_patch + 1, self.emb_dim)
        self.key_emb = torch.nn.Embedding(self.num_patch + 1, self.emb_dim)
        self.value_emb = torch.nn.Identity()
 
    def forward(self, x0):
        v0 = torch.arange(x0.shape[1])
        v1 = v0.unsqueeze(0)
        v2 = v1.unsqueeze(-1)
        v3 = v0.unsqueeze(-1)
        v4 = v3 + v2
        v5 = v4.reshape(-1)
        q = self.query_emb(v5)
        k = self.key_emb(v5)
        v = self.value_emb(v5)
        v6 = q @ k.transpose(-2, -1)
        v7 = v6 / self.pos_dim**0.5
        v8 = F.softmax(v7, dim=-1)
        v9 = F.dropout(v8, 0.0)
        d_v9 = v9 @ v
        return d_v9 

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.tensor([[1], [4], [7], [2], [-2], [6], [3], [5]])
