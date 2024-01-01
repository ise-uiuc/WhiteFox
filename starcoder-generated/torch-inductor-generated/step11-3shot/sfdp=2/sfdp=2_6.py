
class Model(torch.nn.Module):
    def __init__(self, input_hidden_size):
        super(Model, self).__init__()
        self.dropout_p = 0.9
        self.key = torch.zeros((input_hidden_size, input_hidden_size))
        self.query = torch.zeros((input_hidden_size, input_hidden_size))
        self.value = torch.zeros((input_hidden_size, input_hidden_size))
        self.input_hidden_size = input_hidden_size
        self.inv_scale_factor = 0.01
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, input):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
input_hidden_size = 6
m = Model(input_hidden_size)

# Inputs to the model
input1 = torch.randn((3, 6))
