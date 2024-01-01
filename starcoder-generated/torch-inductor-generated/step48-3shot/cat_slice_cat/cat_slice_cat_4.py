
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, size):
        c1 = torch.cat([x1, x2, x3, x4, x5])
        i1 = torch.tensor([70273749298880, 38956906369248, 16316086777680, 83297495521792, 191839786542528,
                           376992761456332, 221880851359940, 0, -16781096230092, -27847728347500,
                           -98222995813580, 0, 793685538262556])
        i2 = torch.tensor([65279, 44612, 19652, 83839, 17440, 34485, 151729, 0, -20462, -38108, -123384, 0, 561049])
        c2 = i1 * c1
        b1 = c2[:, -170141183460469231731687303715884105728]
        l1 = c2[:, i2.tolist()]
        a1 = torch.cat([b1, l1], dim=1)
        return l1[:, :size]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
x2 = torch.randn(1, 7, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
x4 = torch.randn(1, 9, 64, 64)
x5 = torch.randn(7, 10, 64, 64)
