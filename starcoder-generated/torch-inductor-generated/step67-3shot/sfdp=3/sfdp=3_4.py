
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=p)
 
  def forward(self, input, dropout_p=0.5):
        input_matrix = torch.matmul(input, input.transpose(-2, -1))
        scale = math.sqrt(input.shape[-1])
        scaled_matrix = input_matrix * scale
        softmax_matrix = self.softmax(scaled_matrix)
        dropout_matrix = self.dropout(softmax_matrix)
        output = torch.matmul(dropout_matrix, input)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
