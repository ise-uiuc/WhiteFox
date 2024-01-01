
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.inv_scale_factor = 1 / (self.dropout_p * 9)
        self.linear_0 = torch.nn.Linear(128, 128, False)
        self.linear_1 = torch.nn.Linear(128, 128, False)
        self.linear_2 = torch.nn.Linear(256, 128, False)
        self.linear_out = torch.nn.Linear(128, 256, False)

    def forward(self, query, key, value, mask):
        r1 = self.linear_0(query)
        r2 = self.linear_1(key)
        r3 = self.linear_2(value)
        qk = torch.matmul(r1, r2.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        v1 = dropout_qk.matmul(r3)
        v2 = self.linear_out(v1)
        output = v2 + mask.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 6, 128)
key = torch.randn(1, 50, 128)
value = torch.randn(1, 50, 128)
mask = torch.randn(2, 50, 128)
