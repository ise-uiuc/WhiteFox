
class Model(torch.nn.Module):
    def __init__(self, d_model, dropout_p):
        super().__init__()
        self.inv_scale_factor = 1.0 / (d_model ** 0.5)
        self.dropout_p = dropout_p

    def forward(self, x1, x2, x3):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
d_model = 256
dropout_p = 0.1
m = Model(d_model, dropout_p)

# Inputs to the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
