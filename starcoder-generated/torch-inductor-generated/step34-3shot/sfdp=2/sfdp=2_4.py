
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(float(qk.size(-1))))
        scaled_qk = torch.div(qk, inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = (dropout_qk.matmul(value),)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 512)
key = torch.randn(1, 16, 128)
value = torch.randn(1, 16, 128)
