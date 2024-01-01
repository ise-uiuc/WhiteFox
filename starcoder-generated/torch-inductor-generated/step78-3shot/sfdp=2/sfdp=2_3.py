
class Model(torch.nn.Module):
    def __init__(self, num_head=1, dropout_p=0.1):
        super(Model, self).__init__()
        self.linear_query = torch.nn.Linear(32, 8)
        self.linear_key = torch.nn.Linear(16, 8)
        self.linear_value = torch.nn.Linear(16, 8)
        self.linear_output = torch.nn.Linear(8, 1)
        self.num_head = num_head
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, mask=None):
        bsz = query.size(0)
        query = query.contiguous().view(bsz, self.num_head, -1).transpose(1, 2)
        key = key.contiguous().view(bsz, self.num_head, -1).transpose(1, 2)
        value = value.contiguous().view(bsz, self.num_head, -1).transpose(1, 2)
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)
        attn = torch.bmm(query, key.transpose(1, 2))
        inv_scale_factor = 1.0 / math.sqrt(float(query.size(-1)))
        attn = attn * inv_scale_factor
        attn = torch.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn * mask
        attn = torch.nn.functional.dropout(attn, p=self.dropout_p)
        attn = torch.bmm(attn, value)
        attn = attn.contiguous().transpose(1, 2).contiguous().view(bsz, -1)
        output = self.linear_output(attn)
        return output, attn

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 4, 32)
key = torch.randn(2, 8, 16)
value = torch.randn(2, 8, 16)
mask = torch.zeros(2, 4, 8).bernoulli_(1 - 0.5)
