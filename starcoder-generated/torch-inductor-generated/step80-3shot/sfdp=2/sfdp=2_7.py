
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 4, 10, 30)
key = torch.randn(1, 4, 20, 60)
value = torch.randn(1, 4, 20, 60)
dropout_p = 0.2
inv_scale_factor = 0.25
