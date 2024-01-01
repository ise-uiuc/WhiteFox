
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = torch.tensor(1.0 / np.sqrt(key.shape[-1]), dtype=query.dtype, device=query.device)
        softmax_qk = (qk * scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m1 = Model1()

# Inputs to the model
value = torch.randn(1, 8, 23, 100)
key = torch.randn(1, 8, 23, 200)
query = torch.randn(1, 8, 20, 200)
