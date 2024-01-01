
class Model(torch.nn.Module):
    def __init__(self, dropout_p, input_size, num_heads, head_size, output_size, scale_factor):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x1, x2):
        q = self.linear1(x1)
        k = self.linear1(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
input_size = 128
num_heads = 4
head_size = 64
output_size = 256
scale_factor = 1.0
dropout_p = 0.8
m = Model(dropout_p, input_size, num_heads, head_size, output_size, scale_factor)

# Inputs to the model
x1 = torch.randn(1024, 128)
x2 = torch.randn(1024, 128)
