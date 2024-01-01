
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def scaled_dot_product(self, query, key, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        return scaled_qk
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p=0.2):
        scaled_qk = self.scaled_dot_product(query, key, inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 4, 8)
key = torch.randn(1, 3, 8, 4)
value = torch.randn(1, 3, 8, 4)
inv_scale_factor = torch.randn(1)
