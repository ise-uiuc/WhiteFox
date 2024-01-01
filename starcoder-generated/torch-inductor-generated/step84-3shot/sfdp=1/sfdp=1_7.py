
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(32, 32, bias=None)
 
    def forward(self, x1, x2):
        qkv = self.qkv(x1)
        query, key, value = qkv.split(32, dim=-1)
        inv_scale_factor = 1 / math.sqrt(key.shape[-1])
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32)
x2 = torch.randn(1, 8, 32)
