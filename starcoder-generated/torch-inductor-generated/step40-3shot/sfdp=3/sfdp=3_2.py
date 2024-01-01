
class Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

        self.query_proj = torch.nn.Linear(d_model, d_model)
        self.key_proj = torch.nn.Linear(d_model, d_model)
        self.value_proj = torch.nn.Linear(d_model, d_model)
        self.scaled_proj = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x1):
        q = self.query_proj(x1)
        k = self.key_proj(x1)
        v = self.value_proj(x1)
        scale_factor = self.scaled_proj.exp()
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(128, 4, 0.1)

# Inputs to the model
x1 = torch.randn(1, 64, 128)
