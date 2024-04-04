Instruction *InstCombinerImpl::visitMul(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  if (Value *V =
          simplifyMulInst(Op0, Op1, I.hasNoSignedWrap(), I.hasNoUnsignedWrap(),
                          SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // Also allow combining multiply instructions on vectors.
  {
    Value *NewOp;
    Constant *C1, *C2;
    const APInt *IVal;
    if (match(&I, m_Mul(m_Shl(m_Value(NewOp), m_Constant(C2)),
                        m_Constant(C1))) &&
        match(C1, m_APInt(IVal))) {
      // ((X << C2)*C1) == (X * (C1 << C2))
      Constant *Shl = ConstantExpr::getShl(C1, C2);
      BinaryOperator *Mul = cast<BinaryOperator>(I.getOperand(0));
      BinaryOperator *BO = BinaryOperator::CreateMul(NewOp, Shl);
      if (HasNUW && Mul->hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (HasNSW && Mul->hasNoSignedWrap() && Shl->isNotMinSignedValue())
        BO->setHasNoSignedWrap();
      return BO;
    }    if (match(&I, m_Mul(m_Value(NewOp), m_Constant(C1)))) {
      // Replace X*(2^C) with X << C, where C is either a scalar or a vector.
      if (Constant *NewCst = ConstantExpr::getExactLogBase2(C1)) {
        BinaryOperator *Shl = BinaryOperator::CreateShl(NewOp, NewCst);

        if (HasNUW)
          Shl->setHasNoUnsignedWrap();
        if (HasNSW) {
          const APInt *V;
          if (match(NewCst, m_APInt(V)) && *V != V->getBitWidth() - 1)
            Shl->setHasNoSignedWrap();
        }

        return Shl;
      }
    }
  }

  return Changed ? &I : nullptr;
}

