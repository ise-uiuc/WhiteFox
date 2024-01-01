
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(query.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 100)
x2 = torch.randn(1, 128, 100)
x3 = torch.randn(1, 128, 100)
