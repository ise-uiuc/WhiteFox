ing
class Model(torch.nn.Module):
    def my_split(self, x):
        q1 = F.relu(x)
        q2 = x
        q3 = F.leaky_relu(x)
 
        q = torch.cat([q1, q2, q3], dim=1)
        return torch.split(q, 1, dim=1)
 
    def forward(self, x1):
        split_tensors = self.my_split(x1)
        x2 = torch.cat([split_tensors[i][0] for i in range(3)], dim=1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
