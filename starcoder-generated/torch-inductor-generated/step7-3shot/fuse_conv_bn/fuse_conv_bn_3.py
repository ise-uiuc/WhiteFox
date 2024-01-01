
class Model(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(F.max_pool2d(F.pad(x, (1,1,1,1), value=0.3), (3,5), stride=(1,1))), torch.sigmoid(x) + 1], 1)
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
