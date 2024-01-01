
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def attention(self, query, key, value):
        inv_scale = 1.0 / math.sqrt(key.size(-1))
        return torch.matmul(query, 
                key.transpose(-2, -1)) * inv_scale \
                  .softmax(-1) \
                  .matmul(value)

    def forward(self, input1, input2):
        v1 = self.attention(input1, input2, input2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(2, 16, 512, 4)
input2 = torch.randn(2, 16, 512, 64)
