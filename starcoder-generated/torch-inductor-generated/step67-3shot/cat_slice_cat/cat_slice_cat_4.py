
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def func(self, x1):
        tensor1 = torch.cat([x1]*3, dim=1)
        tensor2 = tensor1[:, 0:9223372036854775807]
        tensor3 = tensor2[:, 0:320]
        tensor4 = torch.cat([tensor1, tensor3], dim=1)
        return tensor4
 
    def forward(self, x):
        for x in x:
            x = self.func(x)
        return x.sum()

# Initializing the model
m_list = [torch.rand(1, 320) for i in range(2)]
m = Model()

# Inputs to the model
x = m_list
