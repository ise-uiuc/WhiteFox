bool llvm::simplifyLoop(Loop *L, DominatorTree *DT, LoopInfo *LI,
                        ScalarEvolution *SE, AssumptionCache *AC,
                        MemorySSAUpdater *MSSAU, bool PreserveLCSSA) {
  bool Changed = false;

  while (!Worklist.empty())
    Changed |= simplifyOneLoop(Worklist.pop_back_val(), Worklist, DT, LI, SE,
                               AC, MSSAU, PreserveLCSSA);

  return Changed;
}

static bool simplifyOneLoop(Loop *L, SmallVectorImpl<Loop *> &Worklist,
                            DominatorTree *DT, LoopInfo *LI,
                            ScalarEvolution *SE, AssumptionCache *AC,
                            MemorySSAUpdater *MSSAU, bool PreserveLCSSA) {
  bool Changed = false;
  if (MSSAU && VerifyMemorySSA)
    MSSAU->getMemorySSA()->verifyMemorySSA();

  // Check to see that no blocks (other than the header) in this loop have
  // predecessors that are not in the loop.  This is not valid for natural
  // loops, but can occur if the blocks are unreachable.  Since they are
  // unreachable we can just shamelessly delete those CFG edges!
  for (BasicBlock *BB : L->blocks()) {
    if (BB == L->getHeader())
      continue;    SmallPtrSet<BasicBlock*, 4> BadPreds;
    for (BasicBlock *P : predecessors(BB))
      if (!L->contains(P))
        BadPreds.insert(P);

    // Delete each unique out-of-loop (and thus dead) predecessor.
    for (BasicBlock *P : BadPreds) {

      LLVM_DEBUG(dbgs() << "LoopSimplify: Deleting edge from dead predecessor "
                        << P->getName() << "\n");

      // Zap the dead pred's terminator and replace it with unreachable.
      Instruction *TI = P->getTerminator();
      LLVM_DEBUG(dbgs() << "Instruction *TI = P->getTerminator();\n");
      changeToUnreachable(TI, PreserveLCSSA,
                          /*DTU=*/nullptr, MSSAU);
      Changed = true;
    }
  }

  // If this loop has multiple exits and the exits all go to the same
  // block, attempt to merge the exits. This helps several passes, such
  // as LoopRotation, which do not support loops with multiple exits.
  // SimplifyCFG also does this (and this code uses the same utility
  // function), however this code is loop-aware, where SimplifyCFG is
  // not. That gives it the advantage of being able to hoist
  // loop-invariant instructions out of the way to open up more
  // opportunities, and the disadvantage of having the responsibility
  // to preserve dominator information.
  auto HasUniqueExitBlock = [&]() {
    BasicBlock *UniqueExit = nullptr;
    for (auto *ExitingBB : ExitingBlocks)
      for (auto *SuccBB : successors(ExitingBB)) {
        if (L->contains(SuccBB))
          continue;        if (!UniqueExit)
          UniqueExit = SuccBB;
        else if (UniqueExit != SuccBB)
          return false;
      }

    return true;
  };
  if (HasUniqueExitBlock()) {
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      BasicBlock *ExitingBlock = ExitingBlocks[i];
      if (!ExitingBlock->getSinglePredecessor()) continue;
      BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
      if (!BI || !BI->isConditional()) continue;
      CmpInst *CI = dyn_cast<CmpInst>(BI->getCondition());
      if (!CI || CI->getParent() != ExitingBlock) continue;

  return Changed;
}

