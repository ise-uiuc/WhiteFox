
class Model(torch.nn.Module):
    def __init__(self, hidden_size: int = 10):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_size, 3))
        self.key = torch.nn.Parameter(torch.randn(hidden_size, 3))
        self.value = torch.nn.Parameter(torch.randn(hidden_size, 3))
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scale_factor = 3.25
        softmax_qk = qk.mul(scale_factor).softmax(dim=-1)
        p1 = softmax_qk.data
        dropout_p = 0.45
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)
        p2 = output.data
        return softmax_qk, dropout_qk, output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 10)
__q1__, __p1__, __out1__ = m(x1)

@title Generate an example of PyTorch model that satisfies all the requirements (3 points)
@title Please run this cell to display the input and the output sample of the model you need to generate (use "__" before and after the names of the variables)
# Inputs to the model
x1 = torch.randn(batch_size, max_length, embed_size)
_________ = m(x1)

# Outputs of the model
print(__out1__.shape)
print(__out1__[0][0])

