
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0

        self.query = torch.nn.Parameter(torch.rand((128, 64)))
        self.key = torch.nn.Parameter(torch.rand((256, 64)))
        self.value = torch.nn.Parameter(torch.rand((256, 64)))

    def forward(self, input1):
        q = self.query.unsqueeze(0)
        k = self.key
        v = self.value

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk / math.sqrt(v.size(-1))
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)

        return output

# Initializing the model
m = Model()

# Inputs to the model
input1 = m(torch.randn(1, 128, 64))
