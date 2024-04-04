static bool simplifyLoopCFG(Loop &L, DominatorTree &DT, LoopInfo &LI,
                            ScalarEvolution &SE, MemorySSAUpdater *MSSAU,
                            bool &IsLoopDeleted) {
  bool Changed = false;

  // Constant-fold terminators with known constant conditions.
  Changed |= constantFoldTerminators(L, DT, LI, SE, MSSAU, IsLoopDeleted);

  if (IsLoopDeleted)
    return true;

  return Changed;
}

static bool constantFoldTerminators(Loop &L, DominatorTree &DT, LoopInfo &LI,
                                    ScalarEvolution &SE,
                                    MemorySSAUpdater *MSSAU,
                                    bool &IsLoopDeleted) {

  if (!EnableTermFolding)
    return false;

  // To keep things simple, only process loops with single latch. We
  // canonicalize most loops to this form. We can support multi-latch if needed.
  if (!L.getLoopLatch())
    return false;

  ConstantTerminatorFoldingImpl BranchFolder(L, LI, DT, SE, MSSAU);
  bool Changed = BranchFolder.run();
  LLVM_DEBUG(dbgs() << "bool Changed = BranchFolder.run();\n");
  IsLoopDeleted = Changed && BranchFolder.foldingBreaksCurrentLoop();
  return Changed;
}

