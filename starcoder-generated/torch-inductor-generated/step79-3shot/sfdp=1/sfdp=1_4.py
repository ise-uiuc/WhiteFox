
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, query, key, value):
        scale_factor = torch.sqrt(torch.tensor(query.shape[-1]).t())
        inv_scale_factor = scale_factor.reciprocal()
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * inv_scale_factor
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = MyModel()

# Inputs to the model
query = torch.randn(4, 8, 24)
key = torch.randn(4, 12, 12)
value = torch.randn(4, 12, 24)
