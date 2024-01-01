
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(288, 64)
        self.fc_1 = torch.nn.Linear(64, 8)
        self.fc_2 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, input1):
        v1 = input1.permute(0, 3, 1, 2)
        v2 = self.fc(v1.reshape(v1.size(0), -1))
        v2 = self.sigmoid(v2)
        v2 = v2.unsqueeze(-1).unsqueeze(-1)
        v3 = v2.repeat((1, 1, v1.size(2), v1.size(3)))
        v3 = v1 * v3
        v3 = torch.sum(v3, dim=(2, 3))
        v2 = torch.sum(v2, dim=(2, 3))
        v4 = self.fc_1(v2)
        v4 = self.sigmoid(v4)
        v4 = self.fc_2(v4)
        return v4
# Inputs to the model
input1 = torch.randn(1, 4, 288, 166)
