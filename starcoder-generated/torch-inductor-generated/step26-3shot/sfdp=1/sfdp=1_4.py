
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = query.squeeze(1)
        self.key = key.squeeze(1)
        self.value = value.squeeze(1)
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model(query, key, value, scale_factor, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 1, 32)
