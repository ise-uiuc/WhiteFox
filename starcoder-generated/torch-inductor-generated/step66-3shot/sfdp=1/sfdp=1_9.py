
class Model(torch.nn.Module):
    def forward(self, x2, x3):
        qk = torch.matmul(x2, x3.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.9)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 12, 1024)
x3 = torch.randn(1, 12, 1024)
