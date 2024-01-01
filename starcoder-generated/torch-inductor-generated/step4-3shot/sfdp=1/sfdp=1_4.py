
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        scaled_qk = query.matmul(key.transpose(-2, -1))
        inv_scale_factor = torch.Tensor(1.0 / np.sqrt(key.size(-1))).to(query.device)
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1).to(query.dtype)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 16, 64)
key = torch.randn(1, 16, 64, 64)
value = torch.randn(1, 16, 64, 64)
