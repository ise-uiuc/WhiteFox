
class Model(torch.nn.Module):
    def concat(self, ins: list):
        o = ins[0]
        for i in range(1, len(ins)):
            o = torch.cat((o, ins[i]), dim=1)
        return o

    def forward(self, x1, x2):
        v1 = self.concat([x1, x2])
        v2 = v1[:, :9223372036854775807]
        v3 = v1[:, :x1.size(2)]
        v4 = self.concat([v1, v3])
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 128, 256)
x2 = torch.randn(1, 32, 128, 256)
