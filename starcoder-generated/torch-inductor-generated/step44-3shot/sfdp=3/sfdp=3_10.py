
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(512, 512, bias=False)
        self.query = nn.Linear(512, 512, bias=False)
        self.value = nn.Linear(512, 512, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):
        key, query, value = self.key(inputs), self.query(inputs), self.value(inputs)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(5)
        softmax_qk = self.softmax(scaled_qk)
        return self.dropout(softmax_qk).matmul(value).transpose(-2, -1)

# Initializing the model
m = Model()

# Inputs to the model
_inputs__ = torch.randn(8, 16, 512)
