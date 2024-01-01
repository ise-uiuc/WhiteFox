
class Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(max_seq_len, d_model // n_heads))
        self.key = torch.nn.Parameter(torch.rand(max_seq_len, d_model // n_heads))
        self.value = torch.nn.Parameter(torch.rand(max_seq_len, d_model // n_heads))
        self.inv_scale_factor = 1 if d_model % n_heads!= 0 else max_seq_len ** (-0.25)

    def forward(self, x1):
        qk = torch.matmul(x1, self.query.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model(d_model=64, n_heads=2, max_seq_len=64 + 1, dropout_p=0.2)

# Input to the model
x1 = torch.randn(1, 64)
