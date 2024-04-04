PreservedAnalyses ConstantMergePass::run(Module &M, ModuleAnalysisManager &) {
  if (!mergeConstants(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

static bool mergeConstants(Module &M) {
  // Find all the globals that are marked "used".  These cannot be merged.
  SmallPtrSet<const GlobalValue*, 8> UsedGlobals;
  FindUsedValues(M.getGlobalVariable("llvm.used"), UsedGlobals);
  FindUsedValues(M.getGlobalVariable("llvm.compiler.used"), UsedGlobals);

  // Iterate constant merging while we are still making progress.  Merging two
  // constants together may allow us to merge other constants together if the
  // second level constants have initializers which point to the globals that
  // were just merged.
  while (true) {
    // Find the canonical constants others will be merged with.
    for (GlobalVariable &GV : llvm::make_early_inc_range(M.globals())) {
      // If this GV is dead, remove it.
      GV.removeDeadConstantUsers();
      if (GV.use_empty() && GV.hasLocalLinkage()) {
        GV.eraseFromParent();
        LLVM_DEBUG(dbgs() << "GV.eraseFromParent();\n");
        ++ChangesMade;
        continue;
      }      if (isUnmergeableGlobal(&GV, UsedGlobals))
        continue;

      // This transformation is legal for weak ODR globals in the sense it
      // doesn't change semantics, but we really don't want to perform it
      // anyway; it's likely to pessimize code generation, and some tools
      // (like the Darwin linker in cases involving CFString) don't expect it.
      if (GV.isWeakForLinker())
        continue;

      // Don't touch globals with metadata other then !dbg.
      if (hasMetadataOtherThanDebugLoc(&GV))
        continue;

      Constant *Init = GV.getInitializer();

      // Check to see if the initializer is already known.
      GlobalVariable *&Slot = CMap[Init];

      // If this is the first constant we find or if the old one is local,
      // replace with the current one. If the current is externally visible
      // it cannot be replace, but can be the canonical constant we merge with.
      bool FirstConstantFound = !Slot;
      if (FirstConstantFound || IsBetterCanonical(GV, *Slot)) {
        Slot = &GV;
        LLVM_DEBUG(dbgs() << "Cmap[" << *Init << "] = " << GV.getName()
                          << (FirstConstantFound ? "\n" : " (updated)\n"));
      }
    }

    // Identify all globals that can be merged together, filling in the
    // SameContentReplacements vector. We cannot do the replacement in this pass
    // because doing so may cause initializers of other globals to be rewritten,
    // invalidating the Constant* pointers in CMap.
    for (GlobalVariable &GV : llvm::make_early_inc_range(M.globals())) {
      if (isUnmergeableGlobal(&GV, UsedGlobals))
        continue;

      // We can only replace constant with local linkage.
      if (!GV.hasLocalLinkage())
        continue;

      Constant *Init = GV.getInitializer();

      // Check to see if the initializer is already known.
      auto Found = CMap.find(Init);
      if (Found == CMap.end())
        continue;

      GlobalVariable *Slot = Found->second;
      if (Slot == &GV)
        continue;

      if (makeMergeable(&GV, Slot) == CanMerge::No)
        continue;

      // Make all uses of the duplicate constant use the canonical version.
      LLVM_DEBUG(dbgs() << "Will replace: @" << GV.getName() << " -> @"
                        << Slot->getName() << "\n");
      SameContentReplacements.push_back(std::make_pair(&GV, Slot));
    }

    // Now that we have figured out which replacements must be made, do them all
    // now.  This avoid invalidating the pointers in CMap, which are unneeded
    // now.
    for (unsigned i = 0, e = SameContentReplacements.size(); i != e; ++i) {
      GlobalVariable *Old = SameContentReplacements[i].first;
      GlobalVariable *New = SameContentReplacements[i].second;
      replace(M, Old, New);
      LLVM_DEBUG(dbgs() << "replace(M, Old, New);\n");
      ++ChangesMade;
      ++NumIdenticalMerged;
    }

    if (ChangesMade == OldChangesMade)
      break;
    OldChangesMade = ChangesMade;

    SameContentReplacements.clear();
    CMap.clear();
  }

  return ChangesMade;
}

