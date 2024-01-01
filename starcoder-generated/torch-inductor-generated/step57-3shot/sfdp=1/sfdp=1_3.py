
class Model(torch.nn.Module):
    def __init__(self, input_size, batch_size, output_size, dropout, seq_len):
        super(Model, self).__init__()
 
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout = dropout
 
        self.query = torch.nn.Linear(input_size, output_size, bias=False)
        self.key = torch.nn.Linear(input_size, output_size, bias=False)
        self.value = torch.nn.Linear(input_size, output_size, bias=False)
 
    def forward(self, x):
        query, key, value = self.query(x).view(1,self.batch_size*seq_len,self.output_size), \
            self.key(x).view(1,self.batch_size*seq_len,self.output_size), \
                self.value(x).view(1,self.batch_size*seq_len,self.output_size)

        dot = torch.matmul(query, key.transpose(1,2))
        dot_scale = dot.div(float(self.output_size))
        dot_softmax = softmax(dot_scale, dim=1)
        dot_dropout = dropout(dot_softmax, 0.2)
        return dot_dropout.matmul(value)

# Initializing the model
m = Model(i, b*s, h, 0.2, s)

# Input to the model
x1 = torch.randn(b*s, i)
