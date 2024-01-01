
class Model(torch.nn.Module):
    def __init__(self, in_features, num_heads, dim_out):
        super().__init__()
        self.query = torch.nn.Linear(in_features, in_features)
        self.key = torch.nn.Linear(in_features, in_features)
        self.value = torch.nn.Linear(in_features, in_features)
        self.out = torch.nn.Linear(in_features, dim_out)
 
    def forward(self, x1):
        q = self.query(x1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=num_heads)
        k = self.key(x1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=num_heads)
        v = self.value(x1)
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.out(output)
# Initializing the model
m = Model(64, 8, 256)

# Inputs to the model
x1 = torch.randn(1, 64, 256)
