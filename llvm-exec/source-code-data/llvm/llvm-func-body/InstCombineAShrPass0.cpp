Instruction *InstCombinerImpl::visitAShr(BinaryOperator &I) {
  if (Value *V = simplifyAShrInst(I.getOperand(0), I.getOperand(1), I.isExact(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();
  const APInt *ShAmtAPInt;
  if (match(Op1, m_APInt(ShAmtAPInt)) && ShAmtAPInt->ult(BitWidth)) {
    unsigned ShAmt = ShAmtAPInt->getZExtValue();    // If the shift amount equals the difference in width of the destination
    // and source scalar types:
    // ashr (shl (zext X), C), C --> sext X
    Value *X;
    if (match(Op0, m_Shl(m_ZExt(m_Value(X)), m_Specific(Op1))) &&
        ShAmt == BitWidth - X->getType()->getScalarSizeInBits())
      return new SExtInst(X, Ty);

    // We can't handle (X << C1) >>s C2. It shifts arbitrary bits in. However,
    // we can handle (X <<nsw C1) >>s C2 since it only shifts in sign bits.
    const APInt *ShOp1;
    if (match(Op0, m_NSWShl(m_Value(X), m_APInt(ShOp1))) &&
        ShOp1->ult(BitWidth)) {
      unsigned ShlAmt = ShOp1->getZExtValue();
      if (ShlAmt < ShAmt) {
        // (X <<nsw C1) >>s C2 --> X >>s (C2 - C1)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShAmt - ShlAmt);
        auto *NewAShr = BinaryOperator::CreateAShr(X, ShiftDiff);
        NewAShr->setIsExact(I.isExact());
        return NewAShr;
      }
      if (ShlAmt > ShAmt) {
        // (X <<nsw C1) >>s C2 --> X <<nsw (C1 - C2)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShlAmt - ShAmt);
        auto *NewShl = BinaryOperator::Create(Instruction::Shl, X, ShiftDiff);
        NewShl->setHasNoSignedWrap(true);
        return NewShl;
      }
    }

    if (match(Op0, m_AShr(m_Value(X), m_APInt(ShOp1))) &&
        ShOp1->ult(BitWidth)) {
      unsigned AmtSum = ShAmt + ShOp1->getZExtValue();
      // Oversized arithmetic shifts replicate the sign bit.
      AmtSum = std::min(AmtSum, BitWidth - 1);
      // (X >>s C1) >>s C2 --> X >>s (C1 + C2)
      return BinaryOperator::CreateAShr(X, ConstantInt::get(Ty, AmtSum));
    }

    if (match(Op0, m_OneUse(m_SExt(m_Value(X)))) &&
        (Ty->isVectorTy() || shouldChangeType(Ty, X->getType()))) {
      // ashr (sext X), C --> sext (ashr X, C')
      Type *SrcTy = X->getType();
      ShAmt = std::min(ShAmt, SrcTy->getScalarSizeInBits() - 1);
      Value *NewSh = Builder.CreateAShr(X, ConstantInt::get(SrcTy, ShAmt));
      return new SExtInst(NewSh, Ty);
    }

    if (ShAmt == BitWidth - 1) {
      // ashr i32 or(X,-X), 31 --> sext (X != 0)
      if (match(Op0, m_OneUse(m_c_Or(m_Neg(m_Value(X)), m_Deferred(X)))))
        return new SExtInst(Builder.CreateIsNotNull(X), Ty);

      // ashr i32 (X -nsw Y), 31 --> sext (X < Y)
      Value *Y;
      if (match(Op0, m_OneUse(m_NSWSub(m_Value(X), m_Value(Y)))))
        return new SExtInst(Builder.CreateICmpSLT(X, Y), Ty);
    }

    // If the shifted-out value is known-zero, then this is an exact shift.
    if (!I.isExact() &&
        MaskedValueIsZero(Op0, APInt::getLowBitsSet(BitWidth, ShAmt), 0, &I)) {
      I.setIsExact();
      return &I;
    }
  }

  return nullptr;
}

