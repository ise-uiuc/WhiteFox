
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.inv_scale_factor = 1 / math.sqrt(64 / 4)

    def forward(self, query, key, value, dropout_p=0.):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Query tensor
query = torch.randn(1, 16, 64)

# Key tensor
key = torch.randn(1, 16, 64)

# Value tensor
value = torch.randn(1, 16, 64)

