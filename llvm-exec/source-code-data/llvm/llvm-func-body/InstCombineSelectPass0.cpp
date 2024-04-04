Instruction *InstCombinerImpl::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  Type *SelType = SI.getType();

  if (auto *FCmp = dyn_cast<FCmpInst>(CondVal)) {
    Value *Cmp0 = FCmp->getOperand(0), *Cmp1 = FCmp->getOperand(1);
    // Are we selecting a value based on a comparison of the two values?
    if ((Cmp0 == TrueVal && Cmp1 == FalseVal) ||
        (Cmp0 == FalseVal && Cmp1 == TrueVal)) {
      // Canonicalize to use ordered comparisons by swapping the select
      // operands.
      //
      // e.g.
      // (X ugt Y) ? X : Y -> (X ole Y) ? Y : X
      if (FCmp->hasOneUse() && FCmpInst::isUnordered(FCmp->getPredicate())) {
        FCmpInst::Predicate InvPred = FCmp->getInversePredicate();
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        // FIXME: The FMF should propagate from the select, not the fcmp.
        Builder.setFastMathFlags(FCmp->getFastMathFlags());
        Value *NewCond = Builder.CreateFCmp(InvPred, Cmp0, Cmp1,
                                            FCmp->getName() + ".inv");
        Value *NewSel = Builder.CreateSelect(NewCond, FalseVal, TrueVal);
        return replaceInstUsesWith(SI, NewSel);
      }
    }
  }

  // Fold (select C, (gep Ptr, Idx), Ptr) -> (gep Ptr, (select C, Idx, 0))
  // Fold (select C, Ptr, (gep Ptr, Idx)) -> (gep Ptr, (select C, 0, Idx))
  auto SelectGepWithBase = [&](GetElementPtrInst *Gep, Value *Base,
                               bool Swap) -> GetElementPtrInst * {
    Value *Ptr = Gep->getPointerOperand();
    if (Gep->getNumOperands() != 2 || Gep->getPointerOperand() != Base ||
        !Gep->hasOneUse())
      return nullptr;
    Value *Idx = Gep->getOperand(1);
    if (isa<VectorType>(CondVal->getType()) && !isa<VectorType>(Idx->getType()))
      return nullptr;
    Type *ElementType = Gep->getResultElementType();
    Value *NewT = Idx;
    Value *NewF = Constant::getNullValue(Idx->getType());
    if (Swap)
      std::swap(NewT, NewF);
    Value *NewSI =
        Builder.CreateSelect(CondVal, NewT, NewF, SI.getName() + ".idx", &SI);
    return GetElementPtrInst::Create(ElementType, Ptr, {NewSI});
  };
  if (auto *TrueGep = dyn_cast<GetElementPtrInst>(TrueVal))
    if (auto *NewGep = SelectGepWithBase(TrueGep, FalseVal, false))
      return NewGep;
  if (auto *FalseGep = dyn_cast<GetElementPtrInst>(FalseVal))
    if (auto *NewGep = SelectGepWithBase(FalseGep, TrueVal, true))
      return NewGep;

  if (SelectInst *TrueSI = dyn_cast<SelectInst>(TrueVal)) {
    if (TrueSI->getCondition()->getType() == CondVal->getType()) {
      // select(C, select(C, a, b), c) -> select(C, a, c)
      if (TrueSI->getCondition() == CondVal) {
        if (SI.getTrueValue() == TrueSI->getTrueValue())
          return nullptr;
        return replaceOperand(SI, 1, TrueSI->getTrueValue());
      }
      // select(C0, select(C1, a, b), b) -> select(C0&C1, a, b)
      // We choose this as normal form to enable folding on the And and
      // shortening paths for the values (this helps getUnderlyingObjects() for
      // example).
      if (TrueSI->getFalseValue() == FalseVal && TrueSI->hasOneUse()) {
        Value *And = Builder.CreateLogicalAnd(CondVal, TrueSI->getCondition());
        replaceOperand(SI, 0, And);
        replaceOperand(SI, 1, TrueSI->getTrueValue());
        return &SI;
      }
    }
  }
  if (SelectInst *FalseSI = dyn_cast<SelectInst>(FalseVal)) {
    if (FalseSI->getCondition()->getType() == CondVal->getType()) {
      // select(C, a, select(C, b, c)) -> select(C, a, c)
      if (FalseSI->getCondition() == CondVal) {
        if (SI.getFalseValue() == FalseSI->getFalseValue())
          return nullptr;
        return replaceOperand(SI, 2, FalseSI->getFalseValue());
      }
      // select(C0, a, select(C1, a, b)) -> select(C0|C1, a, b)
      if (FalseSI->getTrueValue() == TrueVal && FalseSI->hasOneUse()) {
        Value *Or = Builder.CreateLogicalOr(CondVal, FalseSI->getCondition());
        replaceOperand(SI, 0, Or);
        replaceOperand(SI, 2, FalseSI->getFalseValue());
        return &SI;
      }
    }
  }

  return nullptr;
}

