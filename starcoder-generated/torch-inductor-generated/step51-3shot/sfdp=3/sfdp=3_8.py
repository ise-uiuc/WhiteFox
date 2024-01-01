
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.q_proj = torch.nn.Linear(16, 32)
        self.k_proj = torch.nn.Linear(24, 32)
        self.v_proj = torch.nn.Linear(40, 32)
        self.scale_factor = math.sqrt(16 / 576)
        
    def forward(self, query, key, value, padding_mask):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        dot = torch.matmul(q, k.transpose(-2, -1))
        dot = dot * self.scale_factor
        dot_mask = dot.masked_fill(padding_mask, -float('inf'))
        softmax = F.softmax(dot_mask, dim=-1)
        dropout = self.dropout(softmax)
        ouput = torch.matmul(dropout, v)
        return output
    
# Input to the model
query = torch.randn(16, 16)
key = torch.randn(16, 24)
value = torch.randn(16, 40)
padding_mask = torch.empty(16, 16, dtype=torch.bool).bernoulli_(0.5)
