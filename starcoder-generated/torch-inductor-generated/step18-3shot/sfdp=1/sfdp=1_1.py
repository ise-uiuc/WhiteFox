
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)) 
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        output = dropout_qk.matmul(value) 
        return output

# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 2, 20, 30)
key = torch.randn(1, 2, 20, 40)
value = torch.randn(1, 2, 20, 50)
inv_scale_factor = torch.scalar_tensor(0.4, dtype=torch.float)
dropout_p = 0.5
