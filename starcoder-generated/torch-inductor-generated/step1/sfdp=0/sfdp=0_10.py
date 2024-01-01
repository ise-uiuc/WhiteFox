
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        scale = np.sqrt(query.shape[-1])
        inv_scale = 1.0 / scale
        matmul1 = torch.matmul(query, torch.transpose(key, -2, -1))
        div = matmul1.div(inv_scale)
        softmax = div.softmax(dim=-1)
        matmul2 = torch.matmul(softmax, value)
        return matmul2

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 64)
key = torch.randn(1, 128, 64)
value = torch.randn(1, 128, 64)
