
class Model(torch.nn.Module):
    def __init__(self, num_heads, dim_head=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head

    def forward(self, x1):
        q = x1
        k = x1
        v = x1
        q = q.reshape(1, 32, 1, 1)
        k = k.reshape(1, 32, 1, 1)
        v = v.reshape(1, 32, 1, 1)
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        qk = torch.matmul(q, k)
        inv_scale_factor = 1 / math.sqrt(self.dim_head)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(v)
        return output.transpose(-2, -1).reshape(1, 1, 1)

# Initializing the model
m = Model(num_heads=1)

# Inputs to the model
x1 = torch.randn(1, 32, 128, 128)
