
class Model(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scale_factor = math.sqrt(embed_dim)

    def forward(self, q, x1, x2):
        qk = torch.matmul(q, x1.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(16)

# Inputs to the model
q = torch.randn(2, 16)
x1 = torch.randn(2, 16, 16)
x2 = torch.randn(2, 16, 4)
