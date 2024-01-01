
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, __input__):
        qkv = torch.randn(20, 3, 64, 64)
        q, k, v = torch.split(qkv, [count_1, 1, 1], dim=-2)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(q.shape[-1])
        softmax_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = softmax_qk.softmax(dim=-1)
        softmax_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(softmax_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
