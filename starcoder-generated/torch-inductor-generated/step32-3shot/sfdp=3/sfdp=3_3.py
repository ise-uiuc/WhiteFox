
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.key = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)

    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1).squeeze(1)
        scale_factor = q.size(-1) ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        dropout_p = 0.8
        output = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1), p=dropout_p).matmul(self.value(x1))
        return output


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
