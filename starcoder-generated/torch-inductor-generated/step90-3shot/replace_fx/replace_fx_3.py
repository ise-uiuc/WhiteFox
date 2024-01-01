
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        class Submodule(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        class Submodule_2(torch.nn.Module):
            def forward(self, a, b):
                c = a.clone()
                c += b
                return c
        self.submodule_1 = Submodule()
        self.submodule_2 = Submodule_2()
    def forward(self, a, b):
        a = self.submodule_1.forward(a, b)
        b = self.submodule_2.forward(a, b)
        c = F.dropout(a, p=0.5)
        d = F.dropout2d(b)
        return c + d
# Inputs to the model
x = torch.tensor([[0.0220696403748088, 0.5876495242118835, 0.18166532063961029], [0.9331319408416748, 0.19494042181491852, 0.6093977737426758], [0.9239399933815, 0.10546592041730881, 0.7460148525238037], [0.39459258880615234, 0.08337637262868881, 0.36462094183921814]])
x1 = torch.tensor([[0.9492112283706665, 0.035508981704711914, 0.686618800163269], [0.7105578737258911, 0.4893934323310852, 0.9271129608154297], [0.608536274433136, 0.5062210321426392, 0.10266485576629639], [0.7512137937545776, 0.7993564963340759, 0.8088854627513885], [0.22926199374198914, 0.9413532857894897, 0.5397661352157593], [0.5315694546699524, 0.7255492711067199, 0.11795864157247543]])
