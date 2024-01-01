
class Model(torch.nn.Module):
    def __init__(self, input_tensor_shape):
        super().__init__()
        self.hidden_size = input_tensor_shape[0]
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = (self.hidden_size ** -0.5)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(input_tensor_shape=(16, 1, 1, 1))

# Inputs to the model
q = torch.randn(1, 16, 1, 1)
k = torch.randn(1, 1, 1, 16)
v = torch.randn(1, 1, 1, 16)
