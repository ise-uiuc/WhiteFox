
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear =  torch.nn.Linear(20, 30)
 
def forward(self, input_value):
    matmul1 = torch.matmul(input_value, w1)
    matmul2 = torch.matmul(matmul1, w2.transpose(0, 1))
    matmul3 = matmul2.matmul(w3)
    return matmul3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(3, 20)
