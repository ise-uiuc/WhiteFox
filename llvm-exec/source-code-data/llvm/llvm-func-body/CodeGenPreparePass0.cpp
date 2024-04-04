bool CodeGenPrepare::runOnFunction(Function &F) {

  if (skipFunction(F))
    return false;

  if (!DisableBranchOpts) {
    MadeChange = false;
    // Use a set vector to get deterministic iteration order. The order the
    // blocks are removed may affect whether or not PHI nodes in successors
    // are removed.
    SmallSetVector<BasicBlock *, 8> WorkList;
    for (BasicBlock &BB : F) {
      SmallVector<BasicBlock *, 2> Successors(successors(&BB));
      MadeChange |= ConstantFoldTerminator(&BB, true);
      if (!MadeChange)
        continue;      for (BasicBlock *Succ : Successors)
        if (pred_empty(Succ))
          WorkList.insert(Succ);
    }

    // Delete the dead blocks and any of their dead successors.
    MadeChange |= !WorkList.empty();
    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      SmallVector<BasicBlock *, 2> Successors(successors(BB));

      DeleteDeadBlock(BB);

      for (BasicBlock *Succ : Successors)
        if (pred_empty(Succ))
          WorkList.insert(Succ);
    }

    // Merge pairs of basic blocks with unconditional branches, connected by
    // a single edge.
    if (EverMadeChange || MadeChange)
      MadeChange |= eliminateFallThrough(F);

    EverMadeChange |= MadeChange;
  }

  return EverMadeChange;
}

bool llvm::ConstantFoldTerminator(BasicBlock *BB, bool DeleteDeadConditions,
                                  const TargetLibraryInfo *TLI,
                                  DomTreeUpdater *DTU) {
  Instruction *T = BB->getTerminator();
  IRBuilder<> Builder(T);

  // Branch - See if we are conditional jumping on constant
  if (auto *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch    BasicBlock *Dest1 = BI->getSuccessor(0);
    BasicBlock *Dest2 = BI->getSuccessor(1);

    if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Replace the conditional branch with an unconditional one.
      BranchInst *NewBI = Builder.CreateBr(Dest1);

      // Transfer the metadata to the new branch instruction.
      NewBI->copyMetadata(*BI, {LLVMContext::MD_loop, LLVMContext::MD_dbg,
                                LLVMContext::MD_annotation});

      Value *Cond = BI->getCondition();
      BI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      return true;
    }

    if (auto *Cond = dyn_cast<ConstantInt>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getZExtValue() ? Dest1 : Dest2;
      BasicBlock *OldDest = Cond->getZExtValue() ? Dest2 : Dest1;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      OldDest->removePredecessor(BB);

      // Replace the conditional branch with an unconditional one.
      BranchInst *NewBI = Builder.CreateBr(Destination);

      // Transfer the metadata to the new branch instruction.
      NewBI->copyMetadata(*BI, {LLVMContext::MD_loop, LLVMContext::MD_dbg,
                                LLVMContext::MD_annotation});

      BI->eraseFromParent();
      if (DTU)
        DTU->applyUpdates({{DominatorTree::Delete, BB, OldDest}});
      return true;
    }

    return false;
  }

  if (auto *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch to an
    // unconditional branch.
    auto *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *DefaultDest = SI->getDefaultDest();
    BasicBlock *TheOnlyDest = DefaultDest;    // If the default is unreachable, ignore it when searching for TheOnlyDest.
    if (isa<UnreachableInst>(DefaultDest->getFirstNonPHIOrDbg()) &&
        SI->getNumCases() > 0) {
      TheOnlyDest = SI->case_begin()->getCaseSuccessor();
    }

    bool Changed = false;

    // Figure out which case it goes to.
    for (auto It = SI->case_begin(), End = SI->case_end(); It != End;) {
      // Found case matching a constant operand?
      if (It->getCaseValue() == CI) {
        TheOnlyDest = It->getCaseSuccessor();
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (It->getCaseSuccessor() == DefaultDest) {
        MDNode *MD = getValidBranchWeightMDNode(*SI);
        unsigned NCases = SI->getNumCases();
        // Fold the case metadata into the default if there will be any branches
        // left, unless the metadata doesn't match the switch.
        if (NCases > 1 && MD) {
          // Collect branch weights into a vector.
          SmallVector<uint32_t, 8> Weights;
          extractBranchWeights(MD, Weights);

          // Merge weight of this case to the default weight.
          unsigned Idx = It->getCaseIndex();
          // TODO: Add overflow check.
          Weights[0] += Weights[Idx + 1];
          // Remove weight for this case.
          std::swap(Weights[Idx + 1], Weights.back());
          Weights.pop_back();
          SI->setMetadata(LLVMContext::MD_prof,
                          MDBuilder(BB->getContext()).
                          createBranchWeights(Weights));
        }
        // Remove this entry.
        BasicBlock *ParentBB = SI->getParent();
        DefaultDest->removePredecessor(ParentBB);
        It = SI->removeCase(It);
        End = SI->case_end();

        // Removing this case may have made the condition constant. In that
        // case, update CI and restart iteration through the cases.
        if (auto *NewCI = dyn_cast<ConstantInt>(SI->getCondition())) {
          CI = NewCI;
          It = SI->case_begin();
        }

        Changed = true;
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (It->getCaseSuccessor() != TheOnlyDest)
        TheOnlyDest = nullptr;

      // Increment this iterator as we haven't removed the case.
      ++It;
    }

    if (CI && !TheOnlyDest) {
      // Branching on a constant, but not any of the cases, go to the default
      // successor.
      TheOnlyDest = SI->getDefaultDest();
    }

    // If we found a single destination that we can fold the switch into, do so
    // now.
    if (TheOnlyDest) {
      // Insert the new branch.
      Builder.CreateBr(TheOnlyDest);
      BasicBlock *BB = SI->getParent();

      SmallSet<BasicBlock *, 8> RemovedSuccessors;

      // Remove entries from PHI nodes which we no longer branch to...
      BasicBlock *SuccToKeep = TheOnlyDest;
      for (BasicBlock *Succ : successors(SI)) {
        if (DTU && Succ != TheOnlyDest)
          RemovedSuccessors.insert(Succ);
        // Found case matching a constant operand?
        if (Succ == SuccToKeep) {
          SuccToKeep = nullptr; // Don't modify the first branch to TheOnlyDest
        } else {
          Succ->removePredecessor(BB);
        }
      }

      // Delete the old switch.
      Value *Cond = SI->getCondition();
      SI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      if (DTU) {
        std::vector<DominatorTree::UpdateType> Updates;
        Updates.reserve(RemovedSuccessors.size());
        for (auto *RemovedSuccessor : RemovedSuccessors)
          Updates.push_back({DominatorTree::Delete, BB, RemovedSuccessor});
        DTU->applyUpdates(Updates);
      }
      return true;
    }

    if (SI->getNumCases() == 1) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      auto FirstCase = *SI->case_begin();
      Value *Cond = Builder.CreateICmpEQ(SI->getCondition(),
          FirstCase.getCaseValue(), "cond");

      // Insert the new branch.
      BranchInst *NewBr = Builder.CreateCondBr(Cond,
                                               FirstCase.getCaseSuccessor(),
                                               SI->getDefaultDest());
      SmallVector<uint32_t> Weights;
      if (extractBranchWeights(*SI, Weights) && Weights.size() == 2) {
        uint32_t DefWeight = Weights[0];
        uint32_t CaseWeight = Weights[1];
        // The TrueWeight should be the weight for the single case of SI.
        NewBr->setMetadata(LLVMContext::MD_prof,
                           MDBuilder(BB->getContext())
                               .createBranchWeights(CaseWeight, DefWeight));
      }

      // Update make.implicit metadata to the newly-created conditional branch.
      MDNode *MakeImplicitMD = SI->getMetadata(LLVMContext::MD_make_implicit);
      if (MakeImplicitMD)
        NewBr->setMetadata(LLVMContext::MD_make_implicit, MakeImplicitMD);

      // Delete the old switch.
      SI->eraseFromParent();
      return true;
    }
    return Changed;
  }

  if (auto *IBI = dyn_cast<IndirectBrInst>(T)) {
    // indirectbr blockaddress(@F, @BB) -> br label @BB
    if (auto *BA =
          dyn_cast<BlockAddress>(IBI->getAddress()->stripPointerCasts())) {
      BasicBlock *TheOnlyDest = BA->getBasicBlock();
      SmallSet<BasicBlock *, 8> RemovedSuccessors;      // Insert the new branch.
      Builder.CreateBr(TheOnlyDest);

      BasicBlock *SuccToKeep = TheOnlyDest;
      for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
        BasicBlock *DestBB = IBI->getDestination(i);
        if (DTU && DestBB != TheOnlyDest)
          RemovedSuccessors.insert(DestBB);
        if (IBI->getDestination(i) == SuccToKeep) {
          SuccToKeep = nullptr;
        } else {
          DestBB->removePredecessor(BB);
        }
      }
      Value *Address = IBI->getAddress();
      IBI->eraseFromParent();
      if (DeleteDeadConditions)
        // Delete pointer cast instructions.
        RecursivelyDeleteTriviallyDeadInstructions(Address, TLI);

      // Also zap the blockaddress constant if there are no users remaining,
      // otherwise the destination is still marked as having its address taken.
      if (BA->use_empty())
        BA->destroyConstant();

      // If we didn't find our destination in the IBI successor list, then we
      // have undefined behavior.  Replace the unconditional branch with an
      // 'unreachable' instruction.
      if (SuccToKeep) {
        BB->getTerminator()->eraseFromParent();
        new UnreachableInst(BB->getContext(), BB);
      }

      if (DTU) {
        std::vector<DominatorTree::UpdateType> Updates;
        Updates.reserve(RemovedSuccessors.size());
        for (auto *RemovedSuccessor : RemovedSuccessors)
          Updates.push_back({DominatorTree::Delete, BB, RemovedSuccessor});
        DTU->applyUpdates(Updates);
      }
      return true;
    }
  }

  return false;
}

