bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;

  SmallVector<GlobalValue *, 4> UsedV;
  collectUsedGlobalVariables(M, UsedV, /*CompilerUsed=*/false);
  collectUsedGlobalVariables(M, UsedV, /*CompilerUsed=*/true);
  Used.insert(UsedV.begin(), UsedV.end());

  do {
    std::vector<WeakTrackingVH> Worklist;
    Deferred.swap(Worklist);    LLVM_DEBUG(doFunctionalCheck(Worklist));

    LLVM_DEBUG(dbgs() << "size of module: " << M.size() << '\n');
    LLVM_DEBUG(dbgs() << "size of worklist: " << Worklist.size() << '\n');

    // Insert functions and merge them.
    for (WeakTrackingVH &I : Worklist) {
      if (!I)
        continue;
      Function *F = cast<Function>(I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage()) {
        Changed |= insert(F);
      }
    }
    LLVM_DEBUG(dbgs() << "size of FnTree: " << FnTree.size() << '\n');
  } while (!Deferred.empty());

  return Changed;
}

bool MergeFunctions::insert(Function *NewFunction) {
  std::pair<FnTreeType::iterator, bool> Result =
      FnTree.insert(FunctionNode(NewFunction));

  if (Result.second) {
    assert(FNodesInTree.count(NewFunction) == 0);
    FNodesInTree.insert({NewFunction, Result.first});
    LLVM_DEBUG(dbgs() << "Inserting as unique: " << NewFunction->getName()
                      << '\n');
    return false;
  }

  Function *DeleteF = NewFunction;
  mergeTwoFunctions(OldF.getFunc(), DeleteF);
  return true;
}

