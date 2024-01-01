
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * inv_scale_factor1
        softmax_qk = torch.nn.functional.dropout(F.softmax(scaled_qk, dim=-1), p=0.2)
        output = torch.matmul(softmax_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 3, 5)
key = torch.randn(1, 1, 2, 5)
value = torch.randn(1, 1, 2, 5)
inv_scale_factor1 = torch.randn(1)
