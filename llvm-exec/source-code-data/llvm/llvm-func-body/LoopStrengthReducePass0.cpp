PreservedAnalyses LoopStrengthReducePass::run(Loop &L, LoopAnalysisManager &AM,
                                              LoopStandardAnalysisResults &AR,
                                              LPMUpdater &) {
  if (!ReduceLoopStrength(&L, AM.getResult<IVUsersAnalysis>(L, AR), AR.SE,
                          AR.DT, AR.LI, AR.TTI, AR.AC, AR.TLI, AR.MSSA))
    return PreservedAnalyses::all();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

static bool ReduceLoopStrength(Loop *L, IVUsers &IU, ScalarEvolution &SE,
                               DominatorTree &DT, LoopInfo &LI,
                               const TargetTransformInfo &TTI,
                               AssumptionCache &AC, TargetLibraryInfo &TLI,
                               MemorySSA *MSSA) {

  // Remove any extra phis created by processing inner loops.
  Changed |= DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());
  if (EnablePhiElim && L->isLoopSimplifyForm()) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
    SCEVExpander Rewriter(SE, DL, "lsr", false);
#ifndef NDEBUG
    Rewriter.setDebugType(DEBUG_TYPE);
#endif
    unsigned numFolded = Rewriter.replaceCongruentIVs(L, &DT, DeadInsts, &TTI);
    if (numFolded) {
      Changed = true;
      RecursivelyDeleteTriviallyDeadInstructionsPermissive(DeadInsts, &TLI,
                                                           MSSAU.get());
      DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());
    }
  }
  // LSR may at times remove all uses of an induction variable from a loop.
  // The only remaining use is the PHI in the exit block.
  // When this is the case, if the exit value of the IV can be calculated using
  // SCEV, we can replace the exit block PHI with the final value of the IV and
  // skip the updates in each loop iteration.
  if (L->isRecursivelyLCSSAForm(DT, LI) && L->getExitBlock()) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
    SCEVExpander Rewriter(SE, DL, "lsr", true);
    int Rewrites = rewriteLoopExitValues(L, &LI, &TLI, &SE, &TTI, Rewriter, &DT,
                                         UnusedIndVarInLoop, DeadInsts);
    if (Rewrites) {
      Changed = true;
      RecursivelyDeleteTriviallyDeadInstructionsPermissive(DeadInsts, &TLI,
                                                           MSSAU.get());
      DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());
    }
  }

  if (AllowTerminatingConditionFoldingAfterLSR) {
    if (auto Opt = canFoldTermCondOfLoop(L, SE, DT, LI)) {
      auto [ToFold, ToHelpFold, TermValueS, MustDrop] = *Opt;      Changed = true;
      NumTermFold++;

      BasicBlock *LoopPreheader = L->getLoopPreheader();
      BasicBlock *LoopLatch = L->getLoopLatch();

      (void)ToFold;
      LLVM_DEBUG(dbgs() << "To fold phi-node:\n"
                        << *ToFold << "\n"
                        << "New term-cond phi-node:\n"
                        << *ToHelpFold << "\n");

      Value *StartValue = ToHelpFold->getIncomingValueForBlock(LoopPreheader);
      (void)StartValue;
      Value *LoopValue = ToHelpFold->getIncomingValueForBlock(LoopLatch);

      // See comment in canFoldTermCondOfLoop on why this is sufficient.
      if (MustDrop)
        cast<Instruction>(LoopValue)->dropPoisonGeneratingFlags();

      // SCEVExpander for both use in preheader and latch
      const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
      SCEVExpander Expander(SE, DL, "lsr_fold_term_cond");
      SCEVExpanderCleaner ExpCleaner(Expander);

      assert(Expander.isSafeToExpand(TermValueS) &&
             "Terminating value was checked safe in canFoldTerminatingCondition");

      // Create new terminating value at loop header
      Value *TermValue = Expander.expandCodeFor(TermValueS, ToHelpFold->getType(),
                                                LoopPreheader->getTerminator());

      LLVM_DEBUG(dbgs() << "Start value of new term-cond phi-node:\n"
                        << *StartValue << "\n"
                        << "Terminating value of new term-cond phi-node:\n"
                        << *TermValue << "\n");

      // Create new terminating condition at loop latch
      BranchInst *BI = cast<BranchInst>(LoopLatch->getTerminator());
      ICmpInst *OldTermCond = cast<ICmpInst>(BI->getCondition());
      IRBuilder<> LatchBuilder(LoopLatch->getTerminator());
      Value *NewTermCond =
          LatchBuilder.CreateICmp(CmpInst::ICMP_EQ, LoopValue, TermValue,
                                  "lsr_fold_term_cond.replaced_term_cond");
      // Swap successors to exit loop body if IV equals to new TermValue
      if (BI->getSuccessor(0) == L->getHeader())
        BI->swapSuccessors();

      LLVM_DEBUG(dbgs() << "Old term-cond:\n"
                        << *OldTermCond << "\n"
                        << "New term-cond:\b" << *NewTermCond << "\n");

      BI->setCondition(NewTermCond);

      OldTermCond->eraseFromParent();
      DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());

      ExpCleaner.markResultUsed();
    }
  }

  if (SalvageableDVIRecords.empty())
    return Changed;

  for (auto &Rec : SalvageableDVIRecords)
    Rec->clear();
  SalvageableDVIRecords.clear();
  DVIHandles.clear();
  return Changed;
}

