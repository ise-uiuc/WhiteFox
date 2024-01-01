
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 15/1000)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model(12)
 
# Inputs of the model
query = torch.randn(1, 12, 3, 64)
key = torch.randn(1, 6, 12, 64)
value = torch.randn(1, 6, 12, 64)
scale_factor = torch.constant([[0.1]])
