
class MyModule(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
 
    def calculate(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, query, key, value, inv_scale_factor):
        return self.calculate(query, key, value, inv_scale_factor)
 
# Initializing the module
my_module = MyModule(dropout_p=0.0)
 
# Inputs to the module
query = torch.randn(1, 3, 8)
key   = torch.randn(1, 6, 8)
value = torch.randn(1, 6, 2)
inv_scale_factor = torch.tensor(0.0)
