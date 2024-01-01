
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, batch_size):
        super().__init__()
        self.mha_query = torch.nn.Linear(dim, dim)
        self.mha_key = torch.nn.Linear(dim, dim)
        self.mha_value = torch.nn.Linear(dim, dim)
        self.mha_scale_factor = 1 / math.sqrt(dim)
        self.dropout_p = 0.1

    def forward(self, x1):
        batch_size = x1.size(0)
        query = x1
        key = x1.transpose(0, 1)
        value = x1

        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.mha_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(256, 4, 1)

# Inputs to the model
x1 = torch.randn(1, 256, 8, 8)
x2 = torch.randn(256, 256, 8, 8)
x3 = torch.randn(256, 256, 8, 8)
