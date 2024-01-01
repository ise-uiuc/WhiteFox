
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        query = torch.nn.functional.normalize(q)
        key = torch.nn.functional.normalize(k)
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = scaled_qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(value)
        return output
 
# Initializing the model with weights for parameters
m = Model()

# Inputs to the model
q = torch.randn(1, 64, 20)
k = torch.randn(1, 64, 64)
v = torch.randn(1, 64, 64)
scale_factor = 1 / math.sqrt(math.sqrt(64))
dropout_p = 0.5
