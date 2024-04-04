Instruction *InstCombinerImpl::visitAtomicRMWInst(AtomicRMWInst &RMWI) {

  // Volatile RMWs perform a load and a store, we cannot replace this by just a
  // load or just a store. We chose not to canonicalize out of general paranoia
  // about user expectations around volatile.
  if (RMWI.isVolatile())
    return nullptr;

  if (!isIdempotentRMW(RMWI))
    return nullptr;

  // We chose to canonicalize all idempotent operations to an single
  // operation code and constant.  This makes it easier for the rest of the
  // optimizer to match easily.  The choices of or w/0 and fadd w/-0.0 are
  // arbitrary.
  if (RMWI.getType()->isIntegerTy() &&
      RMWI.getOperation() != AtomicRMWInst::Or) {
    RMWI.setOperation(AtomicRMWInst::Or);
    return replaceOperand(RMWI, 1, ConstantInt::get(RMWI.getType(), 0));
  } else if (RMWI.getType()->isFloatingPointTy() &&
             RMWI.getOperation() != AtomicRMWInst::FAdd) {
    RMWI.setOperation(AtomicRMWInst::FAdd);
    return replaceOperand(RMWI, 1, ConstantFP::getNegativeZero(RMWI.getType()));
  }

  return nullptr;
}

