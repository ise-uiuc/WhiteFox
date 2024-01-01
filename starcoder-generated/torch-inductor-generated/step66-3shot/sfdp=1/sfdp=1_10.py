
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.dropout_p = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        # A dropout layer is included but not applied. The dropout behavior is controlled by the probability of dropout_p. The input to the dropout layer is just a constant for simplicity. It will not be used in the inference.
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 8)
x2 = torch.randn(2, 1, 8)
