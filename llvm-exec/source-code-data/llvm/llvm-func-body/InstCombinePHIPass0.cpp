Instruction *InstCombinerImpl::visitPHINode(PHINode &PN) {
  if (Value *V = simplifyInstruction(&PN, SQ.getWithInstruction(&PN)))
    return replaceInstUsesWith(PN, V);

  // If this is a trivial cycle in the PHI node graph, remove it.  Basically, if
  // this PHI only has a single use (a PHI), and if that PHI only has one use (a
  // PHI)... break the cycle.
  if (PN.hasOneUse()) {
    if (foldIntegerTypedPHI(PN))
      return nullptr;    Instruction *PHIUser = cast<Instruction>(PN.user_back());
    if (PHINode *PU = dyn_cast<PHINode>(PHIUser)) {
      SmallPtrSet<PHINode*, 16> PotentiallyDeadPHIs;
      PotentiallyDeadPHIs.insert(&PN);
      if (isDeadPHICycle(PU, PotentiallyDeadPHIs))
        return replaceInstUsesWith(PN, PoisonValue::get(PN.getType()));
    }

    // If this phi has a single use, and if that use just computes a value for
    // the next iteration of a loop, delete the phi.  This occurs with unused
    // induction variables, e.g. "for (int j = 0; ; ++j);".  Detecting this
    // common case here is good because the only other things that catch this
    // are induction variable analysis (sometimes) and ADCE, which is only run
    // late.
    if (PHIUser->hasOneUse() &&
        (isa<BinaryOperator>(PHIUser) || isa<GetElementPtrInst>(PHIUser)) &&
        PHIUser->user_back() == &PN) {
      return replaceInstUsesWith(PN, PoisonValue::get(PN.getType()));
    }
    // When a PHI is used only to be compared with zero, it is safe to replace
    // an incoming value proved as known nonzero with any non-zero constant.
    // For example, in the code below, the incoming value %v can be replaced
    // with any non-zero constant based on the fact that the PHI is only used to
    // be compared with zero and %v is a known non-zero value:
    // %v = select %cond, 1, 2
    // %p = phi [%v, BB] ...
    //      icmp eq, %p, 0
    auto *CmpInst = dyn_cast<ICmpInst>(PHIUser);
    // FIXME: To be simple, handle only integer type for now.
    if (CmpInst && isa<IntegerType>(PN.getType()) && CmpInst->isEquality() &&
        match(CmpInst->getOperand(1), m_Zero())) {
      ConstantInt *NonZeroConst = nullptr;
      bool MadeChange = false;
      for (unsigned I = 0, E = PN.getNumIncomingValues(); I != E; ++I) {
        Instruction *CtxI = PN.getIncomingBlock(I)->getTerminator();
        Value *VA = PN.getIncomingValue(I);
        if (isKnownNonZero(VA, DL, 0, &AC, CtxI, &DT)) {
          if (!NonZeroConst)
            NonZeroConst = getAnyNonZeroConstInt(PN);

          if (NonZeroConst != VA) {
            replaceOperand(PN, I, NonZeroConst);
            MadeChange = true;
          }
        }
      }
      if (MadeChange)
        return &PN;
    }
  }

  return nullptr;
}

