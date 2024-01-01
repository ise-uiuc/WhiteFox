
class M(torch.nn.Module):
    # self.weight = torch.zeros(nChannels, 100)
    def forward(self, x):
        y = torch.sigmoid(self.weight @ self.relu(x))
        return y
# Inputs to the model
x = torch.randn(1, 100)
