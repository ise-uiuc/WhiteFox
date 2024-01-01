
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1000)
 
    def forward(self, tensor):
        v1 = self.linear1(tensor)
        v2 = v1 + self.linear2.weight
        v3 = v2 + self.linear2.bias
        v4 = v3 + self.linear3.weight
        v5 = v4 + self.linear3.bias
        return_tensor = []
        idx = 1
        for i in range(64):
            return_tensor.append(v5[idx])
            idx += 64
        return return_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1024, 512)
