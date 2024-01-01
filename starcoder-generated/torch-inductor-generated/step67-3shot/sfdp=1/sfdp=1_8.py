
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1. / np.sqrt(10.), device=qk.device)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 40, 128)
key = torch.randn(1, 40, 128)
value = torch.randn(1, 40, 128)
