
class Model(torch.nn.Module):
    def __init__(self, query_size):
        super(Model, self).__init__()
        self.query_size = query_size
 
    def forward(self, query, key, value, mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk_float = qk.float()
        if mask is not None:
            qk_float = qk_float.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn_weight = torch.softmax(qk_float, dim=-1)
        output = attn_weight @ value
        return output 

# Initializing the model
m = Model(200)

# Inputs to the model
query = torch.randn(1, 5, 200)
key = torch.randn(1, 7, 200)
value = torch.randn(1, 7, 12)
mask = torch.cat((torch.triu(torch.ones(5, 7), 1), torch.tril(torch.ones(5, 7)).T), 0)
