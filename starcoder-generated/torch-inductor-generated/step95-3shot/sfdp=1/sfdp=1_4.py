
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = math.pow(query.size(-1)).float()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.Softmax(dim=-1)(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.00249854953108063)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 512, 64)
key = torch.randn(1, 512, 512)
value = torch.randn(1, 512, 512)
