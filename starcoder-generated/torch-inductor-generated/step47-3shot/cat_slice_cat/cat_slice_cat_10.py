
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat((x1, x2))
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x2.shape[1]]
        v4 = torch.cat((v1, v3))
        v5 = torch.cat((x3, x4), dim=1)
        v6 = v5[:, 0:9223372036854775807]
        v7 = v6[:, 0:x4.shape[1]]
        v8 = torch.cat((v5, v7), dim=1)
        v9 = torch.cat((v4, v8), dim=0)
        