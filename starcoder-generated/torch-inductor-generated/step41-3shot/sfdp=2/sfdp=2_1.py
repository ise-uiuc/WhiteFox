
class Model(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = math.sqrt(math.sqrt(query.shape[-1]) / math.sqrt(key.shape[-1]))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.mm(value)
        return output

# Initializing the model
m = Model(torch.nn.Identity(5))

# Inputs to the model
query = torch.randn(1, 2, 5)
key = torch.randn(1, 4, 5)
value = torch.randn(1, 4, 7)
