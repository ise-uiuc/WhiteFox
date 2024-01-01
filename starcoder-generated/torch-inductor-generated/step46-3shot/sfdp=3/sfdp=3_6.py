
class Model(torch.nn.Module):
    def __init__(self, emb_dim, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.query = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = torch.nn.Linear(emb_dim, emb_dim, bias=False)
 
    def forward(self, query, key, value):
        q = self.query(query)
        kt = self.key(key).permute(1,0)
        kv = self.value(value)
        qkt = torch.matmul(q, kt)
        scale = qkt.size(-1) ** -0.5
        qkt_scaled = qkt*scale
        dropout_qkt = torch.nn.functional.dropout(qkt_scaled, p=self.dropout_p)
        result = torch.matmul(dropout_qkt, kv)
        return result

# Initializing the model
m = Model(emb_dim=512)

# Inputs to the model
query = torch.randn(1, 19, 512)
key = torch.randn(1, 10, 512)
value = torch.randn(1, 10, 512)
