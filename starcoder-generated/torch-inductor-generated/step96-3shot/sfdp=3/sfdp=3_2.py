
class Model(torch.nn.Module):
    def __init__(self, dim, heads, dropout_p=0.):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim) 
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, attn_mask):
        query = self.q(query)
        key = self.k(key)
        value = value.permute(0, 2, 1).contiguous()
        key = key.permute(0, 2, 3, 1)\
          .contiguous()\
          .view(-1, key.size(1), key.size(2))
        value = value.view(-1, value.size(2), value.size(1))
        scores = torch.matmul(query, key)\
          .view(-1, 
                        key.size(1))
        scores = scores / math.sqrt(scores.size(1))
        log_attn_mask = (attn_mask == False)
        log_attn_mask = log_attn_mask.repeat(1, scores.size(1))
        max_mask = 1e9 if not attn_mask.any() else -1e9 
        scores = torch.where(log_attn_mask, max_mask, scores)
        dropout_scores = self.dropout(scores)
        softmax_scores = softmax(dropout_scores, dim=-1)
        output = torch.matmul(dropout_scores.unsqueeze(dim=-1), value)
        return output.view(query.size(0), 
                                  query.size(1), 
                                  key.size(2))

# Initializing the model
m = Model(128, 4)

# Inputs to the model
query = torch.randn(10, 128)
key = torch.randn(100, 128)
value = value = torch.randn(100)
attn_mask = torch.randn(1, 100)
m(query, key, value, attn_mask)