
class Model(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_dim,
        n_heads
    ):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(n_heads, dim, input_dim))
        self.key = torch.nn.Parameter(torch.randn(n_heads, dim, input_dim))
        self.value = torch.nn.Parameter(torch.randn(n_heads, dim, input_dim))
        self.dropout = torch.nn.Dropout(p=0.7)
 
    def forward(self, x):
        qk = torch.matmul(x, self.key.transpose(-2, -1))
        scaled_qk = qk * (int(x.shape[-1]) ** -0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model(dim=8, input_dim=512, n_heads=4)

# Inputs to the model
x = torch.randn(4, 8, 512)
