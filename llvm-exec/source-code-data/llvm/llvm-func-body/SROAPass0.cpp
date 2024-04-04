PreservedAnalyses SROAPass::runImpl(Function &F, DomTreeUpdater &RunDTU,
                                    AssumptionCache &RunAC) {
  LLVM_DEBUG(dbgs() << "SROA function: " << F.getName() << "\n");
  C = &F.getContext();
  DTU = &RunDTU;
  AC = &RunAC;

  PreservedAnalyses PA;
  if (!CFGChanged)
    PA.preserveSet<CFGAnalyses>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

bool SROAPass::deleteDeadInstructions(
    SmallPtrSetImpl<AllocaInst *> &DeletedAllocas) {
  bool Changed = false;
  while (!DeadInsts.empty()) {
    Instruction *I = dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val());
    if (!I)
      continue;
    LLVM_DEBUG(dbgs() << "Deleting dead instruction: " << *I << "\n");

    ++NumDeleted;
    I->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

