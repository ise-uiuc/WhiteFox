Instruction *InstCombinerImpl::visitXor(BinaryOperator &I) {
  if (Value *V = simplifyXorInst(I.getOperand(0), I.getOperand(1),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // (A | B) ^ (A | C) --> (B ^ C) & ~A -- There are 4 commuted variants.
  // TODO: Loosen one-use restriction if common operand is a constant.
  Value *D;
  if (match(Op0, m_OneUse(m_Or(m_Value(A), m_Value(B)))) &&
      match(Op1, m_OneUse(m_Or(m_Value(C), m_Value(D))))) {
    if (B == C || B == D)
      std::swap(A, B);
    if (A == C)
      std::swap(C, D);
    if (A == D) {
      Value *NotA = Builder.CreateNot(A);
      return BinaryOperator::CreateAnd(Builder.CreateXor(B, C), NotA);
    }
  }

  return nullptr;
}

