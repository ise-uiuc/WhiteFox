
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embed_query = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.embed_key = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.embed_value = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear0 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.linear1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.dropout0 = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.2)
 
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = self.dropout0(v1)
        q = self.head_query(v2)
        k = self.head_key(x1)
        v3 = self.linear1(x1)
        v4 = self.dropout1(v3)
        v5 = self.head_value(v4)
        return v5

# Initializing the model
m = Model(1024, 8)

# Inputs to the model
x1 = torch.randn(8, 1024)
