
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 128)
        self.fc_drop = torch.nn.Dropout(0.2)
 
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        v1 = self.fc(x)
        v1 = self.fc_drop(v1)
        v2 = v1 + iperf_m
        v3 = math_ops.relu(v2)
        return v3
        
# Initializing the model
m = Model()

# Inputs to the model
iperf_m = torch.randn(128)
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
