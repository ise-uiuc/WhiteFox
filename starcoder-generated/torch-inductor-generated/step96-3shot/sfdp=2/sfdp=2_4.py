
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / self.scale_factor
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=True)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dropout_p = 0.5
scale_factor = 4.0
m = Model(dropout_p, scale_factor)

# Inputs to the model
query = torch.randn(8, 16, 16)
key = torch.randn(8, 64, 16)
value = torch.randn(8, 64, 250)
