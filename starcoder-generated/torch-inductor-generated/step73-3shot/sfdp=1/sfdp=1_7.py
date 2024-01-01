
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = torch.nn.Linear(64, 32)
        self.key_proj = torch.nn.Linear(64, 32)
        self.value_proj = torch.nn.Linear(64, 32)
 
    def forward(self, x1, x2, x3):
        q = self.query_proj(x1)
        k = self.key_proj(x2)
        v = self.value_proj(x3)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(k.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)
