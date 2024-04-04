PreservedAnalyses ReassociatePass::run(Function &F, FunctionAnalysisManager &) {
  // Get the functions basic blocks in Reverse Post Order. This order is used by
  // BuildRankMap to pre calculate ranks correctly. It also excludes dead basic
  // blocks (it has been seen that the analysis in this pass could hang when
  // analysing dead basic blocks).
  ReversePostOrderTraversal<Function *> RPOT(&F);

  // Traverse the same blocks that were analysed by BuildRankMap.
  for (BasicBlock *BI : RPOT) {
    assert(RankMap.count(&*BI) && "BB should be ranked.");
    // Optimize every instruction in the basic block.
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;)
      if (isInstructionTriviallyDead(&*II)) {
        EraseInst(&*II++);
      } else {
        OptimizeInst(&*II);
        assert(II->getParent() == &*BI && "Moved to a different block!");
        ++II;
      }    // Make a copy of all the instructions to be redone so we can remove dead
    // instructions.
    OrderedSet ToRedo(RedoInsts);
    // Iterate over all instructions to be reevaluated and remove trivially dead
    // instructions. If any operand of the trivially dead instruction becomes
    // dead mark it for deletion as well. Continue this process until all
    // trivially dead instructions have been removed.
    while (!ToRedo.empty()) {
      Instruction *I = ToRedo.pop_back_val();
      if (isInstructionTriviallyDead(I)) {
        RecursivelyEraseDeadInsts(I, ToRedo);
        MadeChange = true;
      }
    }

    // Now that we have removed dead instructions, we can reoptimize the
    // remaining instructions.
    while (!RedoInsts.empty()) {
      Instruction *I = RedoInsts.front();
      RedoInsts.erase(RedoInsts.begin());
      if (isInstructionTriviallyDead(I))
        EraseInst(I);
      else
        OptimizeInst(I);
    }
  }

  return PreservedAnalyses::all();
}

