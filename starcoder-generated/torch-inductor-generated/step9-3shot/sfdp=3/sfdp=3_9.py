
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p, inplace=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q1, k1):
        q2 = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = q2.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 64, 86)
k1 = torch.randn(1, 310, 86)
