
class Model(torch.nn.Module):
    def forward(self, x1):
        k = x1
        q = x1
        s = 10
        dp = 0.1
        v = x1
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(s)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dp)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 10)
