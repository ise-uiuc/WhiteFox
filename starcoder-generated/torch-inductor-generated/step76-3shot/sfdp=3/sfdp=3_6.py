
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.tensor(2.0))

    def _attention(self, query, key, value, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

    def forward(self, x1):
        v1 = self._attention(x1, x1, x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
