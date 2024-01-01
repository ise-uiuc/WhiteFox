
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor):
        q = torch.randn([1, 6, 5]).transpose(1, 2)
        k = torch.randn([1, 6, 10])
        v = torch.randn([1, 6, 10])
        output = torch.matmul(q.transpose(1, 2), k) / math.sqrt(k.shape[-1])
        output += torch.randn([1, 12, 12])
        output = torch.softmax(output, -1)
        output = torch.softmax(output, -1)
        output = torch.matmul(output, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn([1, 12, 5])
