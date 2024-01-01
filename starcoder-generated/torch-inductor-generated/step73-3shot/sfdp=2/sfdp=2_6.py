
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(1, 2, 3, 4)
key = torch.randn(1, 2, 3, 5)
inv_scale_factor = torch.randn(1)
dropout_p = torch.randn(1)
