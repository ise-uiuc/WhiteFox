
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query):
        key = self.key(query)
        value = self.value(query)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale = torch.sqrt(torch.tensor(key.shape[-2]).float())
        inv_scale_factor = 1.0 / scale
        scaled_qk = qk.mul(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(dropout_qk, value)
        return output

# Inputs to the model
query = torch.randn(1, 8, 64, 96)
