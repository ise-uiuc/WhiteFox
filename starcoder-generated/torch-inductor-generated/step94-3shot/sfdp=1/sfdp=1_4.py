
class Model(torch.nn.Module):
    def __init__(self, dropout_p, inv_scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.1, inv_scale_factor=math.sqrt(1 / 128))

# Inputs to the model
query = torch.randn(1, 32, 128)
key = torch.randn(1, 32, 128)
value = torch.randn(1, 32, 128)
