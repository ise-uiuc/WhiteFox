
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p=0.5):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = np.sqrt(query.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        return torch.nn.functional.dropout(scaled_qk.softmax(dim=-1).matmul(value), p=dropout_p)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 128)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
