
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.inv_scale_factor = float(3.1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.0)

# Inputs to the model
query = torch.randn(1, 64, 8)
key = torch.randn(1, 8, 2)
value = torch.randn(1, 8, 4)
