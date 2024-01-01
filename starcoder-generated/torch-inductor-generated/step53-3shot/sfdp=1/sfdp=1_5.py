
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qk, qk_scaling_factor, query, key, value, p=0.0, is_training=False, epsilon=1e-6):
        input_dtype = query.dtype
        m = torch.matmul(query, key.transpose(-2, -1))
        qk = m / torch_dtype_to_gain[input_dtype]
        qk = qk / qk_scaling_factor
        if is_training:
            qk = torch.nn.functional.dropout(qk, p, True, False)
        else:
            qk = torch.nn.functional.dropout(qk, p, False, False)
        o = torch.matmul(qk, value)
        return o

# Initializing the model
m = Model()

# Inputs to the model
qk = torch.randn(1,64,10)
qk_scaling_factor = torch.randn(1,1)
query = torch.randn(1, 512, 64)
key = torch.randn(1, 512, 1024)
value = torch.randn(1, 512, 1024)
output = m(qk, qk_scaling_factor, query, key, value)

