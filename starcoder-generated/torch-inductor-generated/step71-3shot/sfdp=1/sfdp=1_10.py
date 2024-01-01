
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, query_size=1024, key_size=1024, dropout_p=0):
        super().__init__()
        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.dropout = dropout_p
        self.W_query = torch.nn.Linear(query_size, num_heads * query_size)
        self.W_key = torch.nn.Linear(key_size, num_heads * query_size)
        self.W_value = torch.nn.Linear(key_size, num_heads * query_size)
        self.W_out = torch.nn.Linear(num_heads * query_size, query_size)
 
    def forward(self, query, key, value):
        inv_sqrtp = torch.sqrt(torch.tensor([query.size()[-1]]).type_as(query)).to(query.device)
        inv_scale_factor = 1 / inv_sqrtp
        logits = torch.matmul(query, key.transpose(-2, -1))
        logits = logits.div(inv_scale_factor)
        weights = logits.softmax(dim=-1)
        weights = torch.nn.functional.dropout(weights, p=self.dropout)
        return torch.matmul(weights, value)

# Initializing the model
m = Model(num_heads=8, query_size=1024, key_size=1024, dropout_p=0)

# Inputs to the model
query = torch.randn(1, 8, 1024)
key = torch.randn(1, 16, 1024)
value = torch.randn(1, 16, 1024)
