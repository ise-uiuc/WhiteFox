
class SelfAttention(torch.nn.Module):
    def __init__(self, query_len: int, key_len: int, value_len: int, dropout_p: float = 0.5):
        super().__init__()
        self.scale_factor = torch.tensor(1 / key_len)
        self.dropout_p = dropout_p
 
        self.W_V = torch.nn.Linear(32, 32)
        self.W_K = torch.nn.Linear(32, 32)
        self.W_Q = torch.nn.Linear(32, 32)
        self.W_O = torch.nn.Linear(32, 32)
 
    def compute_attention(self, query, key, value):
        q1 = self.W_Q(query)
        k1 = self.W_K(key)
        v1 = self.W_V(value)
 
        q2 = q1.view(q1.shape[0], 1, q1.shape[1])
        k2 = k1.view(k1.shape[0], k1.shape[1], 1)
 
        q3 = q2 * k2
 
        s = q3.mean(dim=2)
        s = s.mul(self.scale_factor)
        s = s.exp()
        s = s.mean(dim=1)
 
        d2 = 1 - s
        d1 = 1 - torch.nn.functional.dropout(d2, p=self.dropout_p)
        m = v1 * d1.view(d1.size(0), 1, 1)
        o = m.mean(dim=1)
 
        return self.W_O(o)
 
    def forward(self, query, key, value):
        query_len = query.shape[-2]
        key_len = key.shape[-2]
        value_len = value.shape[-2]
 
        attention = self.compute_attention(query, key, value)
 
        return attention
 
m = SelfAttention(32, 32, 32)

# Inputs to the model
query = torch.randn(8, 32, 32)
key = torch.randn(8, 64, 32)
value = torch.randn(8, 64, 32)
