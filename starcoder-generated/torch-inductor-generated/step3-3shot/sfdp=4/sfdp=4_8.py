
class Model(torch.nn.Module):
    def __init__(self, attention_mask = None, transpose = False):
        super().__init__()
        self.attention_mask = attention_mask
    def forward(self, x2, x3):
        if self.attention_mask is None:
            v7 = x2 @ x3.transpose(-2, -1)
        else:
            v67 = (x2 + x3) * self.attention_mask(x3.size(), x2.dtype)
            v7 = x2 @ v67.transpose(-2, -1)
        v8 = softmax(v7, dim=-1)
        output = v8 @ x3
        return output
    
# Initializing the model
def attention_mask(size, dtype):
    v65 = torch.tril(torch.ones(*size))
    return torch.Tensor(v65)

m = Model(transpose = False)

x2 = torch.randn(5,1,64,64)
x3 = torch.randn(5,1,64,64)
