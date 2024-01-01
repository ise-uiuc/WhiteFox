
class MyModule(nn.Module):
    def forward(self, x):
        v = x.view(x.size(0), -1)
        return np.dot(v, v.t())
