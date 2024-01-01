
def matmul(query, key):
    return torch.matmul(query, key.transpose(-2, -1))

class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = matmul(query, key)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        # `torch.nn.functional.dropout` applies dropout to the input tensor:
        #   - If `p=0` or `inplace=True`, dropout is disabled.
        #   - If `p>0` and `inplace=False`, dropout is enabled.
        #   - If `p>0` and `inplace=True`, dropout is enabled in-place (requires_grad is unchanged).
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
query = torch.randn(1, 1, 128)
key = torch.randn(1, 1, 256)
value = torch.randn(1, 1, 256)
scale_factor = torch.randn(1, 128, 256)
dropout_p = torch.tensor([0.5])
m = Model(query, key, value, scale_factor, dropout_p)

# Inputs to the model
# NOTE: the `torch.ones` function produces a tensor with all ones.
# The `requires_grad` property of tensors will be changed to True by default, and
# the input tensor of `torch.nn.functional.dropout` will also require gradient. 
x1 = torch.ones(1, 1, 128, requires_grad=True)
x2 = torch.ones(1, 1, 256, requires_grad=True)
x3 = torch.ones(1, 1, 256, requires_grad=True)
x4 = torch.ones(1, 128, 256, requires_grad=True)
x5 = torch.ones(1, requires_grad=True)
