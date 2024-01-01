
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 32, 32)
key = torch.randn(1, 16, 32, 32)
value = torch.randn(1, 16, 32, 32)
__inv_scale_factor__ = 1.0 / math.sqrt(0.3)
