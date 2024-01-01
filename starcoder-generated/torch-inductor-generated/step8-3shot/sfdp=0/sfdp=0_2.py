
class Model(torch.nn.Module):
    def __init__(self, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.scale = np.power(self.attention_size, -0.5)
 
    def forward(self, input, values):
        q2q = input[:, 0:self.attention_size]
        k2k = input[:, self.attention_size:]
        q2k2k = values
        scaled_dot_product = torch.matmul(q2q, k2k.transpose(-2, -1)) / self.scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(q2k2k)
        return output

# Initializing the model
m = Model(attention_size=128)

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(10, 256, 1024)
