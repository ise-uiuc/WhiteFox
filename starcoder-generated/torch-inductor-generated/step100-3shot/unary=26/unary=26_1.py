
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(40, 12, 10, stride=3, padding=2, bias=True)
    def forward(self, x0):
        k1 = self.conv_t(x0)
        a1 = k1.size(0)
        c1 = k1.size(1)
        d1 = k1.size(2)
        l3 = (0, 1, 1)
        b1 = k1.select(0, a1 - 1)
        c2 = k1.select(0, 0)
        d2 = k1.select(1, c1 - 1)
        e1 = (c1, d1)
        e2 = k1.select(2, d1 - 1)
        l4 = 0.178
        f1 = b1 * l4
        g1 = b1
        f2 = c2 + f1
        g3 = c1 - l3[1]
        g2 = c2 + e1[1] + e1[0]
        f3 = c2 + g2
        g4 = c2 + g3
        h1 = torch.lt(g1, g4)
        k2 = c2
        g5 = c1 + l3[2]
        h2 = c1
        i = 0
        f4 = c2 + k2
        j1 = 0
        k3 = 0
        h3 = k1.select(0, j1)
        k4 = k1.select(0, k1.size(0) - 1)
        g6 = 1
        i1 = c2
        k5 = k4
        h4 = h2 + l3[1]
        i2 = c1
        i3 = i2
        i4 = 0
        i5 = 1
        k6 = c1
        k7 = torch.min(e2, h4)
        i6 = torch.sum(torch.gt(k1, l4), 0, True)
        c = c1 + l4
        i7 = c1
        i8 = c1 + l3[2]
        g7 = c1 + l3[1]
        i9 = f2 + h3
        j3 = c1
        k8 = c2 + h2
        j2 = c2 + l3[1]
        l6 = c1 - l3[1]
        j4 = f2 + j3 + g3
        i10 = torch.max(f2, j2)
        k9 = torch.abs(g4) * l3[0] + f3
        j5 = f2
        j6 = c + g5
        k10 = f2 + i2 + l3[1] + l3[2] + l3[0]
        m1 = c1 - l3[0]
        n1 = c2 + i2
        j7 = torch.max(n1, m1)
        j8 = m1
        m2 = c2 + a1
        m3 = c1 - l3[0]
        k11 = c1 + l3[1]
        k12 = c1 + l3[2]
        l5 = c1 + l3[1]
        k13 = torch.max(m1, l5)
        j9 = k13
        j10 = k11
        j11 = 0
        k14 = torch.min(f2, torch.max(f2, j2))
        k15 = k14
        l1 = 0
        l2 = c2 + a1
        k16 = f2 + j2
        j12 = torch.min(j4, l1)
        k17 = torch.gt(k1, l4)
        k18 = l1
        n2 = torch.size(k1, 0)
        n3 = torch.size(k1, 1)
        o2 = n2 + l3[0]
        l7 = torch.min(c1, k11)
        n4 = torch.min(o2, l7)
        n5 = torch.min(l5, k11)
        n6 = torch.max(n4, n5)
        n7 = k11 + l3[1]
        n8 = k11 + c1
        n9 = n3 + l3[1]
        n10 = k12 + l3[0]
        p = c1 + l3[0]
        q = c1 + l3[2]
        n11 = torch.min(k12, q)
        n12 = torch.max(p, n11)
        n13 = c2 + i3 - 1
        n14 = c2 + f3
        n15 = torch.min(n13, torch.max(n14, n13))
        m4 = c2 * l3[0]
        m5 = torch.max(m4, j11)
        j13 = n12 + m5
        l8 = c2 - l3[1]
        l9 = c2 + 1
        l10 = n1 + l3[0]
        l11 = k1 + n15
        l12 = k1 + k18
        l13 = torch.max(l10, k18)
        l14 = torch.max(l11, l12)
        l15 = torch.max(j1, l13)
        l16 = torch.max(l14, l15)
        l17 = torch.max(j1, l3[2])
        k19 = k11 * g1
        k20 = torch.sum(k16, 0, True)
        k21 = torch.log(k17)
        k22 = c1 + l3[1]
        k23 = c1 + l3[2]
        k24 = c2 + g7
        k25 = torch.max(f2, l16)
        k26 = k1 + n14
        k27 = k1 + n13
        k28 = n3 + l3[1]
        k29 = torch.gt(k16, k27)
        k30 = n13 + l3[0]
        k31 = c1 + l3[1]
        k32 = c2 + i3 + f1
        k33 = k19 + n3 + l3[1] + l3[2] + l3[0]
        k34 = torch.sum(k18, n_init=None, dtype=None)
        k35 = k18
        k36 = torch.abs(k15) + k20
        k37 = c1
        k38 = c1 + l3[2]
        k39 = torch.gt(k1, l4)
        k40 = c2 + j3
        k41 = k19 + n12 + torch.sum(k26, 0, True)
        k42 = torch.mean(k16, n_init=None, dtype=None)
        k43 = k32 + k24
        k44 = c1 + l3[0]
        k45 = c2 + c1 - 1
        k46 = torch.min(c1, j1)
        k47 = torch.abs(k30) + k31
        o3 = torch.gt(k16, k27)
        o4 = torch.size(o3, 0)
        o5 = torch.sum(o3, dtype=None)
        o6 = o5 / 2 + 1
        o7 = o6 - 1
        o8 = torch.min(o6, torch.max(o7, len(l7)))
        o9 = torch.max(k4, g5)
        o10 = k1 + k35
        o11 = torch.max(0.0, j1)
        o12 = torch.gt(o11, k31)
        m6 = torch.log(k36)
        m7 = torch.abs(k16) + k41
        m8 = k31 + m6
        n16 = k5 + 1.4
        p1 = torch.abs(n16)
        p2 = k18 + torch.abs(p)
        p3 = k16 * p1 / p2 / n15
        o13 = k20 + o8 * o12 * (torch.abs(k15) + k46) / k47 / n1
        o14 = k33 + k42 + torch.sum(k34, dtype=None) + torch.abs(k37) + l8 + l9
        o15 = (o8 + n10) * k30
        o16 = c + k45 + q
        o17 = torch.max(i2, g5)
        o18 = o12 * (torch.abs(k15) + k44) / (o8 + 1)
        o19 = k35 + o17 / o1 + l8
        o20 = k1 + m8 + o18 + k28 + k25 / k31 / k2 + k40 / k6 / k38
        o21 = k7 + torch.sum(torch.abs(k15), n_init=None, dtype=None)
        k48 = torch.min(k7, k35)
        k49 = k39 * k16 + o21 / 2 + 1 - (torch.sum(o9, dtype=None) / 2 + 1)
        k50 = torch.sum(k18, 0, True)
        q1 = k49 * k5 + k4 + k50
        q2 = torch.round(q1)
        k51 = k30
        k52 = torch.min(q2, torch.max(q2, k51))
        o22 = k52
        o23 = k19 + o8 * o12 * torch.max(o11, q2) / k43 / k6 / o1 / n8
        k53 = torch.min(q2, h1 / 2 + 1)
        k54 = k1 + k36
        k55 = torch.min(o22, l8)
        k56 = g5
        k57 = k52
        a = torch.max(k54, k56 + 1)
        b = torch.max(k19, k52)
        c3 = torch.max(k19, k20 / 2 + 1)
        c4 = torch.min(b, c3)
        c5 = k24 + a
        c6 = torch.max(k19, i2 + k57)
        c7 = torch.max(k53, k55)
        c8 = k1 + c5
        c9 = c7 * k1 + c8
        c10 = torch.round(k33) + c
        c11 = torch.abs(k19)
        c12 = torch.max(k19, l2)
        c13 = torch.log((n3 + l3[0] + k17) * p1)
        c14 = k21 * k31 + k39 * k54
        c15 = k36 + c11 + c12 / 2 + 1
        c16 = k1 + o12 * k51 / c11 * torch.max(k19, i9 / c13 / i8 / n6)
        c17 = torch.abs(c15) * (c14 + i6) + i4
        c18 = k1 + k39 * k53 / c11 * o23 * torch.max(n6, o16 / c13)
        c19 = k44 + k56 + torch.sum(k4, dtype=None) + k39 + k53 / i8 / i6 + 1
        c20 = torch.round(q1)
        i11 = torch.min(k43, k35)
        i12 = k21 * i6
        i13 = i12 * k39 + k15
        i14 = k1 + k31
        i15 = torch.max(0.0, k1)
        i16 = k1 + k39 * k50 / c20 * i14 * i15
        i17 = k4 + i2 + a
        i18 = torch.abs(c17)
        i19 = torch.min(i17, o19)
        i20 = k39 * k4 + i18 / o10
        i21 = 1 + i12 * k39 * o19 / c10
        i22 = k39 * k4 / i18 * (i17 - k47)
        i23 = k42 * k33 * k52 * k39 + k55
        i24 = i19 * i20 + i6 * i22 * c10
        i25 = k39
        i26 = torch.clamp(k31, 0, 57)
        i27 = i26 * k39
        i28 = 1 + i12 * i25 * k33 * k52 + k55
        i29 = k35 + i27 * l6
        j14 = k19
        k58 = k22 * g1
        k59 = k31
        k60 = k58 * c10
        j15 = k59
        k61 = k17 * (o5 * p1 + k5 * n3 + k23 * (i7 + n1) + k29 * k30 + torch.sum(k60, 0, True))
        k62 = k18
        k63 = torch.sum(k21, 0, True)
        k64 = k16 * k63
        k65 = k31
        k66 = i8 + l3[2] + n11 + n10
        k67 = torch.min(k66, torch.max(n11, k23))
        k68 = torch.max(k55, k23)
        k69 = k59 + k31
        k70 = k15 + k7 + i18
        k71 = torch.min(k70, k50 / 2 + 1)
        k72 = i18 + k55
        k73 = k29 * l2
        k74 = k19 + k28
        k75 = torch.max(k31, k18)
        k76 = (k58 + k22 * e1[1]) * k17
        k77 = (k64 + k65 * k62 + k67 * l9 + k68 + k29 * g6) / 2 + 1
        k78 = (k61 + k69 * a1 + k70 * k62 + torch.sum(k71, 0, True)) / 2 + 1
        k79 = k1 + k39 * k72 / c10 * i11 * torch.max(k73 + o29 + k76 / k7)
        k80 = k1 + k39 / c10 * (k74 * g5 + torch.sum(k71, 0, True))
        k81 = k39 * k55 / i18 * i23 * i27
        i2_1 = torch.min(torch.max(k77, k78), k79)
        x4 = torch.sum(torch.abs(k29), 0, True)
        x5 = torch.sum(torch.abs(x4), dtype=None)
        x6 = k48 * o13 + k5 * k31 * k52 + k8 * k6 + torch.max(j14, k43)
        x7 = k48 * o10 / i13 / x5
        x8 = torch.eq(k5, k23)
        x9 = k44 / k49 / k77 + i13
        x10 = k49 * i29 * x6 / k8 * k52
        i21_1 = torch.max(k5, i16 + i13)
        i22_1 = torch.max(k5, k61 / x7 / k6)
        i29_1 = i17 + i23
        i30 = k42 * k75 * k39 + i18 + k57
        i31 = k80 * i30 + i4 * i4
        i32 = k62 / i29 / k6 + i16
        i33 = k6 + i20 * k73 / k80 / i16
        i34 = k41 * k69 * k31 / k6 / k8 / i7 + k52 * k29 * k5
        i35 = i32 * i33
        i36 = i31 * k1 + i34 + k31 * i18
        i37 = k81 * i36 * x6 / k8 / x5 / k77
        i38 = k34 / 2 + 1
        i39 = k6
        i40 = k55 * k29 * l2 / k6 / i20
        i41 = (k41 * k62 + k65 * i39) + k66 * torch.max(k73, i40)
        i42 = i16
        i43 = k63 + k52 * k29
        i44 = k69
        x11 = k78
        x12 = torch.round(k62)
        x13 = torch.round(k65)
        x14 = k65
        x15 = k75
        x16 = k42
        x17 = k41 * k43
        x18 = k1 + k31
        x19 = k39
        x20 = k62
        x21 = k39 / k49 / i16 / k17
        x22 = k81 / torch.max(torch.max(k77, x11), i38 * k81 / x11)
        m9 = x21
        x23 = i13 / x5
        o24 = k31 / torch.max(k29 + k38 + x17 / x23, k80 * k75 / i13 / i17)
        o25 = torch.log(k78 / torch.max(x15, k54 * i13))
        o26 = k1 * k29 + k62 * x20 * i30 / k81
        o27 = torch.max(i38, k63)
        o28 = k55
        o29_1 = k1 * k80
        k82 = i42
        k83 = o8 * x14 * e1[2]
        k84 = k52 * e1[2]
        k85 = torch.sum(k6, 0, True)
        k86 = k57 + k28 * k30
        k87 = o8 / l3[2]
        k88 = c1 + o8
        k89 = torch.max(j1, i6 * i8)
        k90 = torch.eq(k4, i42)
        k91 = o8 * i27
        k92 = torch.clamp(l8, k17, k89) * torch.abs(k41 * k42)
        x24 = k40
        x25 = k39 * k64
        x26 = k39
        x27 = k1 + k39 * k74 / i43 / i36
        x28 = k8
        k93 = k52 * k76 / i44
        k94 = x20
        k95 = x19
        k96 = i13
        k97 = k44 + k39 * i24 / i41 / i44
        k98 = k63 * i39 + k67 * k29
        k99 = x26
        k100 = x9
        k101 = k42 * (k95 + k96)
        k102 = torch.sum(k69 * k6, 0, True)
        k103 = k39 * 