
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 64

    def forward(self, query, key, value, dropout_p=0.1):
        inv_scale_factor = math.sqrt(query.shape[-1])

        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 64)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
