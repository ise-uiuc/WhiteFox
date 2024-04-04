PreservedAnalyses LoopDeletionPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &Updater) {

  // If we can prove the backedge isn't taken, just break it and be done.  This
  // leaves the loop structure in place which means it can handle dispatching
  // to the right exit based on whatever loop invariant structure remains.
  if (Result != LoopDeletionResult::Deleted)
    Result = merge(Result, breakBackedgeIfNotTaken(&L, AR.DT, AR.SE, AR.LI,
                                                   AR.MSSA, ORE));

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

breakBackedgeIfNotTaken(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                        LoopInfo &LI, MemorySSA *MSSA,
                        OptimizationRemarkEmitter &ORE) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");

  auto *BTCMax = SE.getConstantMaxBackedgeTakenCount(L);
  if (!BTCMax->isZero()) {
    auto *BTC = SE.getBackedgeTakenCount(L);
    if (!BTC->isZero()) {
      if (!isa<SCEVCouldNotCompute>(BTC) && SE.isKnownNonZero(BTC))
        return LoopDeletionResult::Unmodified;
      if (!canProveExitOnFirstIteration(L, DT, LI))
        return LoopDeletionResult::Unmodified;
    }
  }
  ++NumBackedgesBroken;
  breakLoopBackedge(L, DT, SE, LI, MSSA);
  LLVM_DEBUG(dbgs() << "breakLoopBackedge(L, DT, SE, LI, MSSA);\n");
  return LoopDeletionResult::Deleted;
}

