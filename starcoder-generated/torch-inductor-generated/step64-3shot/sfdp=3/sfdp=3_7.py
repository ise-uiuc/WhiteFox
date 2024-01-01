
class Model(torch.nn.Module):
    def __init__(self, num_heads=2, d_model=64):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.h_dim = d_model // num_heads

        self.W_Q = torch.nn.Linear(512, self.d_model)
        self.W_K = torch.nn.Linear(512, self.d_model)
        self.W_V = torch.nn.Linear(512, self.d_model)
        self._scale_factor = math.sqrt(self.h_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, query, key, value) -> torch.Tensor:
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self._scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

num_heads = 2
d_model = 512

# Inputs to the model
query = torch.randn(1, 10, 512)
key = torch.randn(1, 10, 512)
value = torch.randn(1, 10, 512)
