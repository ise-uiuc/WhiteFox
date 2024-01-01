
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(0.01, dtype=qk.dtype, device=qk.device).sqrt()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_pkl = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return torch.matmul(dropout_qk, value)
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 5, 100, 64)
key = torch.randn(1, 5, 200, 64)
value = torch.randn(1, 5, 200, 64)
