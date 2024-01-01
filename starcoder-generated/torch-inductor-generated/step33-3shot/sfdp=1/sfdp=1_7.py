
class Model(torch.nn.Module):
    def __init__(self):
        self.dropout_p = 0.1
        super().__init__()
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 4)
key = torch.randn(2, 4, 5)
value = torch.randn(2, 4, 5)
scale_factor = torch.randn(2, 1, 1)
