
class Model_2(torch.nn.Module):
    def __init__(self, num_heads, head_size, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
num_heads, head_size, dropout_p = 4, 32, 0.5
m = Model_2(num_heads, head_size, dropout_p)

# Inputs to the model
d_model = [4096, 4096, 4096, 4096]
x1 = torch.randn(2, d_model[0], head_size)
x2 = torch.randn(2, d_model[1], head_size)
x3 = torch.randn(2, d_model[2], head_size)
inv_scale_factor = torch.tensor([d_model[0]] * num_heads)
