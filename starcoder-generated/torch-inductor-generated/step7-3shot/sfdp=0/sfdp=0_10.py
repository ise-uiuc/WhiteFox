
class Attention(torch.nn.Module):
    def __init__(self, q, k, v):
        super(Attention, self).__init__()
        self.m = torch.nn.Linear(1 * q, k)
        self.q = q
        self.k = k
        self.v = v
 
    def forward(self, inputs):
        batch, c_input, dim_input1, dim_input2 = inputs.size()
        outputs = self.m(inputs)
        outputs = outputs.view(batch, self.k, self.q, dim_input1, dim_input2)
        outputs = outputs.transpose(2, 3)
        outputs = outputs.transpose(1, 2)
        outputs = outputs.view(batch * self.q, self.k * dim_input1 * dim_input2, 1)
        return outputs
 
class Model(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(Model, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.attention1 = Attention(self.dim_input, self.dim_hidden, self.dim_hidden)
        self.attention2 = Attention(self.dim_hidden, self.dim_hidden, 1)
 
    def forward(self, x1):
        inputs1 = self.attention1(x1)
        inputs2 = self.attention2(inputs1)
        return inputs2
 
batch = 4
dim_input = 512
dim_hidden = 2048
x1 = torch.randn(batch, 1, dim_input)
