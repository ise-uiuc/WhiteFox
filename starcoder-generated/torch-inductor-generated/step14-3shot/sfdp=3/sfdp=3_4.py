
class Model(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 128 / self.num_heads
        self.scale_factor = self.head_dim**-0.5
        self.query = torch.nn.Parameter(torch.randn(shape[1], self.num_heads, shape[2], shape[2]))
        self.key = torch.nn.Parameter(torch.randn(shape[1], self.num_heads, shape[2], shape[2]))
        self.value = torch.nn.Parameter(torch.randn(shape[1], self.num_heads, shape[2], shape[2]))
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return dropout_qk.matmul(self.value)

# Initializing the model
m = Model((1, 8, 64, 64))

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
