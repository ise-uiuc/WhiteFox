
class Model(torch.nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.query_linear = torch.nn.Linear(input_size, input_size)
        self.key_linear = torch.nn.Linear(input_size, input_size)
        self.value_linear = torch.nn.Linear(input_size, input_size)
        self.num_heads = num_heads

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, x.size(-1))
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        q = self.query_linear(x1).unsqueeze(0)  # [1, 2, 4, 4]
        k = x2
        v = x2
        batch_size = x1.size(0)
        q, k, v = self.map(batch_size, q, k, v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        q, k, v = self.map(batch_size, q, k, v)
        return torch.matmul(q, k.transpose(-2, -1))

# Initializing the model
m = Model(input_size=4, num_heads=2)

# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
