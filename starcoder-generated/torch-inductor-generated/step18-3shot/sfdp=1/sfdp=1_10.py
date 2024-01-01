
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 10)
key = torch.randn(1, 3, 10)
value = torch.randn(1, 3, 5)
inv_scale_factor = 1.0/math.sqrt(3.0)
