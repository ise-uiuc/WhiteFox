
class Model(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p=0.5, scale_value=None):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dropout_p = dropout_p
        self.scale_value = scale_value
        if self.scale_value is None:
            self.scale_value = hidden_size ** 0.5
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        qkv = self.qkv_proj(x1)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q *= self.scale_value
        k *= self.scale_value
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_value)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(hidden_size=64)
m

# Inputs to the model
x1 = torch.randn(1, 8, 64)
x2 = torch.randn(1, 8, 64)
