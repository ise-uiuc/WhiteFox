
class Model(torch.nn.Module):
    def __init__(self, num_heads=12):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(num_heads=12)

# Inputs to the model
q = torch.randn(1, 12, 64, 64)
k = torch.randn(1, 12, 256, 64)
v = torch.randn(1, 12, 256, 64)
scale_factor = torch.randn(12)
dropout_p = 0.2
