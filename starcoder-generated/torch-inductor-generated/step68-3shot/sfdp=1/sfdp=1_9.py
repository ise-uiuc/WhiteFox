
class Model(torch.nn.Module):
    def __init__(self, num_heads, input_size, dropout_p, d_model=128):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.input_size = input_size
        self.dropout_p = dropout_p

        self.query = torch.nn.Parameter(torch.Tensor(num_heads, input_size, d_model))
        self.key = torch.nn.Parameter(torch.Tensor(num_heads, input_size, d_model))
        self.value = torch.nn.Parameter(torch.Tensor(num_heads, input_size, d_model))
        self.inv_scale_factor = torch.nn.Parameter(torch.Tensor())
        torch.nn.init.xavier_uniform_(self.query)
        torch.nn.init.xavier_uniform_(self.key)
        torch.nn.init.xavier_uniform_(self.value)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.inv_scale_factor, 0, 1)

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        return scaled_qk

# Initializing the model
num_heads = 8
dropout_p = 0.2
input_size = 128
d_model = 128
m = Model(num_heads, input_size, dropout_p, d_model)

# Inputs to the model, where, q is the query tensor, k is the key tensor, and v is the value tensor
q = torch.randn(1, num_heads, input_size, d_model)
k = torch.randn(1, num_heads, input_size, d_model)
v = torch.randn(1, num_heads, input_size, d_model)
