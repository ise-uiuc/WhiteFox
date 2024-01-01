
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, query_size, dropout_p=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.query_size = query_size
        self.dropout_p = dropout_p
        self.key = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.inv_scale_factor = torch.nn.Parameter(torch.Tensor([input_size ** -.5]))
        self.query = torch.nn.Parameter(torch.Tensor(output_size, query_size))
        self.value = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        torch.nn.init.normal_(self.key, 0, input_size**-.5)
        torch.nn.init.constant_(self.inv_scale_factor, 1.)
        torch.nn.init.normal_(self.query, 0, input_size**-.5)
        torch.nn.init.normal_(self.value, 0, input_size**-.5)
  
    def forward(self, x1, x2):
        qk = torch.tensordot(x1, self.key, dims=1)
        scaled_qk = qk * (self.inv_scale_factor**-1)
        softmax_qk = torch.nn.functional.dropout(self.softmax(scaled_qk), p=self.dropout_p)
        output = torch.tensordot(softmax_qk, self.value, dims=1)
        return output

# Initializing the model
m = Model(input_size=300, output_size=400, query_size=500)

# Inputs to the model
x1 = torch.randn(500, 300)
x2 = torch.randn(300, 500)
