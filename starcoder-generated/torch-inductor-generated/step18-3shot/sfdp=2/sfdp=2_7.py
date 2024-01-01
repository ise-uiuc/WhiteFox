
class Model(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self._output_linear = torch.nn.Linear(1024, 1000)
 
    def forward(self, qk, softmax_qk, value):
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        self._output_linear(output)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1024)
key   = torch.randn(100, 1024)
value = torch.randn(100, 1024)

