
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.scale_factor = math.sqrt(66 / 33)

    def forward(self, query, key, value, dropout_p):
        k_channels = key.shape[0]
        q = torch.matmul(query, key.transpose(-2, -1))
        scaled_q = q.mul(self.scale_factor)
        softmax_q = scaled_q.softmax(dim=-1)
        softmax_d = torch.nn.functional.dropout(softmax_q, p=dropout_p)
        out = torch.matmul(softmax_d, value)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(66, 33)
y = torch.randn(33, 66)
input = torch.randn(2, 66, 33)
dropout_p = 0.2
