
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0 / (math.sqrt(query.size(-1)) * key.size(-1))
 
    def forward(self, query, key, value, **kwargs):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=kwargs.get('dropout_p', 0.0))
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
