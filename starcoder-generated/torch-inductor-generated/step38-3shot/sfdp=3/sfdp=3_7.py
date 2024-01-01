
class Model(torch.nn.Module):
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.scale_factor = d_model ** -0.5

    def forward(self, query, key, value, dropout_p=0.3):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        y = dropout_qk.matmul(value)
        return y

# Initializing the model
m = Model(d_model=700)

# Inputs to the model
query = torch.randn(1, 1, 700)
key = torch.randn(1, 1, 700)
value = torch.randn(1, 1, 700)
