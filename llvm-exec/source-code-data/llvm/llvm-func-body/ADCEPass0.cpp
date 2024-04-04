ADCEChanged AggressiveDeadCodeElimination::performDeadCodeElimination() {
  initialize();
  markLiveInstructions();
  return removeDeadInstructions();
}

ADCEChanged AggressiveDeadCodeElimination::removeDeadInstructions() {
  ADCEChanged Changed;
  // Updates control and dataflow around dead blocks
  Changed.ChangedControlFlow = updateDeadRegions();

  // The inverse of the live set is the dead set.  These are those instructions
  // that have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : llvm::reverse(instructions(F))) {
    // Check if the instruction is alive.
    if (isLive(&I))
      continue;    if (auto *DII = dyn_cast<DbgInfoIntrinsic>(&I)) {
      // Avoid removing a dbg.assign that is linked to instructions because it
      // holds information about an existing store.
      if (auto *DAI = dyn_cast<DbgAssignIntrinsic>(DII))
        if (!at::getAssignmentInsts(DAI).empty())
          continue;
      // Check if the scope of this variable location is alive.
      if (AliveScopes.count(DII->getDebugLoc()->getScope()))
        continue;

      // Fallthrough and drop the intrinsic.
    } else {
      Changed.ChangedNonDebugInstr = true;
    }

    // Prepare to delete.
    Worklist.push_back(&I);
    LLVM_DEBUG(dbgs() << "Worklist.push_back(&I);\n");
    salvageDebugInfo(I);
  }

  return Changed;
}

