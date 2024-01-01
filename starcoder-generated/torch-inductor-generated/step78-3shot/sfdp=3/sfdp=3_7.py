
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.key = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.value = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2).transpose(-2, -1)
        v = self.value(x2)

        qk = torch.matmul(q, k)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
