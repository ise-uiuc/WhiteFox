static bool runOnBasicBlock(BasicBlock &BB) {
  bool Changed = false;
  for (Instruction &Inst : make_early_inc_range(BB)) {
    if (FenceInst *FI = dyn_cast<FenceInst>(&Inst))
      Changed |= LowerFenceInst(FI);
    else if (AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(&Inst))
      Changed |= lowerAtomicCmpXchgInst(CXI);
    else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(&Inst))
      Changed |= lowerAtomicRMWInst(RMWI);
    else if (LoadInst *LI = dyn_cast<LoadInst>(&Inst)) {
      if (LI->isAtomic())
        LowerLoadInst(LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(&Inst)) {
      if (SI->isAtomic())
        LowerStoreInst(SI);
    }
  }
  return Changed;
}

bool llvm::lowerAtomicRMWInst(AtomicRMWInst *RMWI) {
  IRBuilder<> Builder(RMWI);
  Value *Ptr = RMWI->getPointerOperand();
  Value *Val = RMWI->getValOperand();

  LoadInst *Orig = Builder.CreateLoad(Val->getType(), Ptr);
  Value *Res = buildAtomicRMWValue(RMWI->getOperation(), Builder, Orig, Val);
  Builder.CreateStore(Res, Ptr);
  RMWI->replaceAllUsesWith(Orig);
  RMWI->eraseFromParent();
  return true;
}

