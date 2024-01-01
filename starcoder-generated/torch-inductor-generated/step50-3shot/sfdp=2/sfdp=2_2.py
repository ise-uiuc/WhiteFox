
class Model(torch.nn.Module):
    def __init__(self, dropout_p = 0.5):
        super().__init__()
        self.scale_factor = math.sqrt(1.0 / 128)
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
__input1__ = torch.randn(1, 128, 512)
__input2__ = torch.randn(1, 128, 512)
__input3__ = torch.randn(1, 128, 512)
