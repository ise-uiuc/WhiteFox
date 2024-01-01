
class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
 
    def _generate_parameter(self):
        return torch.randn(self.input_size, self.input_size)    
 
    def forward(self, x1, x2, x3):
        p1 = self._generate_parameter()
        p2 = self._generate_parameter()
        v1 = torch.matmul(x1, p1)
        v2 = torch.matmul(x2, p2)
        v3 = torch.matmul(v1, torch.transpose(v2, -2, -1))
        v4 = v3.div(self.scale_factor)
        v5 = torch.nn.Softmax(dim=-1)(v4)
        v6 = torch.nn.functional.dropout(v5, p=self.dropout_p)
        v7 = torch.matmul(v6, x3)
        return v7

# Initializing the model
seed = 4
torch.manual_seed(seed)

input_size=1024
m = Model(input_size)

# Inputs to the model
x1 = torch.randn(1, input_size)
x2 = torch.randn(1, input_size)
x3 = torch.randn(1, input_size)
