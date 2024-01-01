
class Model(torch.nn.Module):
    def __init__(self, num_heads=16, batch_size=64, sequence_length=64, hidden_size=512, p=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        num_heads, hidden_size_per_head = hidden_size // num_heads, hidden_size // num_heads
        self.key = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.value = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.query = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.inv_scale_factor = (hidden_size_per_head ** -0.25) ** 0.5
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x1):
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = self.dropout(softmax_qk).matmul(self.value)
        return output

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 16, 512)
