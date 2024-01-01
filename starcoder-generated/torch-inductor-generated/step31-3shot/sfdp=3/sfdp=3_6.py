
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor=1./(10**2), dropout_p=.2):
        super().__init__()
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        self.register_buffer('dropout_qk', dropout_qk)
        self.register_buffer('value', value)
 
    def forward(self, x):
        scaled_qk = torch.matmul(x, self.dropout_qk.transpose(-2, -1))
        output = scaled_qk.matmul(self.value)
        return output

# Inplace operation on constant
a = torch.tensor([1, 2, 3], requires_grad=True)
b = torch.tensor([1.2, 2.3, 3.4], requires_grad=True)
c = b.clone()
b_id = id(b)
c_id = id(c)
b += 1
c += float(1)
print('a:', a, 'b:', b, 'c:', c)

# Inputs to the model
x = torch.randn(1, 1, 16, 8)
query = torch.randn(1, 1, 16, 8)
key = torch.randn(1, 1, 8, 8)
value = torch.randn(1, 1, 16, 8)
