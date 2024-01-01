
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, inv_scale_factor=1.0 / math.sqrt(1024)):
        super().__init__()
        # These attributes are required
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
m = Model(dropout_p=0.5, inv_scale_factor=1.0 / math.sqrt(1024))

# Inputs to the model
query = torch.randn(1, 1024, 8, 8)
key = torch.randn(1, 64, 8, 8)
value = torch.randn(1, 64, 8, 8)
