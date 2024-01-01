
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Embedding(100, 32)
        self.key = torch.nn.Embedding(100, 32)
        self.value = torch.nn.Embedding(100, 32)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, queries, keys, values, scale_factor):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor.unsqueeze(-1))
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randint(0, 100, (64, 16)).float()
keys = torch.randint(0, 100, (64, 16)).float()
values = torch.randint(0, 100, (64, 16)).float()
scale = torch.arange(1, 1 + 32 * 32).reshape(1, 32, 32).float()
