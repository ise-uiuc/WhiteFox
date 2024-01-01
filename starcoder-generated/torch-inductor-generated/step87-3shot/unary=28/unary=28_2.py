
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 80)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu(v1)
        v3 = F.softmax(v1, dim=-1)
        v4 = F.sigmoid(v1)
        v5 = F.softplus(v1)
        v6 = F.tanh(v1)
        v7 = F.relu6(v1)
        v8 = torch.nn.functional.l1_loss(v1, v1)
        v9 = torch.nn.functional.mse_loss(v1, v1)
        v10 = torch.nn.functional.adaptive_avg_pool2d(v1, 1)
        v11 = torch.nn.functional.adaptive_max_pool2d(v1, 1)
        v12 = torch.nn.functional.avg_pool2d(v1, 1)
        v13 = torch.nn.functional.max_pool2d(v1, 1)
 
        return v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13

# Initializing the model
m = Model()

# Inputs to the model. The input shape will be (minibatch size, number of input features, 1, number of timesteps)
x1 = torch.randn(1, 40, 1, 2)
