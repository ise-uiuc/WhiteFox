
class SelfAttention(torch.nn.Module):
    def __init__(self, dim, heads, dropout_p=0.1):
        super().__init__()
        self.heads = heads
        self.scale_factor = dim ** -0.5
        self.to_query = torch.nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.to_key = torch.nn.Linear(dim, dim, bias=False)
        self.to_value = torch.nn.Linear(dim, dim, bias=False)
        self.unify = torch.nn.Linear(dim, dim, bias=False)
 
    def forward(self, x):
        queries = self.to_query(x).chunk(self.heads, dim=-1)
        keys = self.to_key(x).chunk(self.heads, dim=-1)
        values = self.to_value(x).chunk(self.heads, dim=-1)
        softmax_qk_outputs = []
        for query, key in zip(queries, keys):
            scaled_qk = torch.matmul(query, key.transpose(-2, -1))
            scaled_qk = scaled_qk / self.scale_factor
            softmax_qk = scaled_qk.softmax(dim=-1)
            dropout_qk = self.dropout(softmax_qk)
            softmax_qk_outputs.append(dropout_qk)
        output = torch.cat([torch.matmul(attention_weight, value) for attention_weight, value in zip(softmax_qk_outputs, values)], dim=-1)
        output = self.unify(output)
        return output

# Initializing the model
m = SelfAttention(128, 8)

# Inputs to the model
__input__ = torch.randn(1, 128, 128)
