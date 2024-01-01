
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query1, key2, value3, scale_factor4, dropout_p5):
        qk = torch.matmul(query1, key2.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor4)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p5)
        output = dropout_qk.matmul(value3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query1 = torch.randn(1, 64, 50, 80)
key2 = torch.randn(1, 8, 50, 80)
value3 = torch.randn(1, 8, 50, 80)
scale_factor4 = torch.tensor(1.0)
dropout_p5 = torch.tensor(0.1)
