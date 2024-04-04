PreservedAnalyses DSEPass::run(Function &F, FunctionAnalysisManager &AM) {
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  bool Changed = eliminateDeadStores(F, AA, MSSA, DT, PDT, AC, TLI, LI);

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MemorySSAAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}

static bool eliminateDeadStores(Function &F, AliasAnalysis &AA, MemorySSA &MSSA,
                                DominatorTree &DT, PostDominatorTree &PDT,
                                AssumptionCache &AC,
                                const TargetLibraryInfo &TLI,
                                const LoopInfo &LI) {
  bool MadeChange = false;

  MSSA.ensureOptimizedUses();
  DSEState State(F, AA, MSSA, DT, PDT, AC, TLI, LI);
  // For each store:
  for (unsigned I = 0; I < State.MemDefs.size(); I++) {
    MemoryDef *KillingDef = State.MemDefs[I];
    if (State.SkipStores.count(KillingDef))
      continue;
    Instruction *KillingI = KillingDef->getMemoryInst();    std::optional<MemoryLocation> MaybeKillingLoc;
    if (State.isMemTerminatorInst(KillingI)) {
      if (auto KillingLoc = State.getLocForTerminator(KillingI))
        MaybeKillingLoc = KillingLoc->first;
    } else {
      MaybeKillingLoc = State.getLocForWrite(KillingI);
    }

    if (!MaybeKillingLoc) {
      LLVM_DEBUG(dbgs() << "Failed to find analyzable write location for "
                        << *KillingI << "\n");
      continue;
    }
    MemoryLocation KillingLoc = *MaybeKillingLoc;
    assert(KillingLoc.Ptr && "KillingLoc should not be null");
    const Value *KillingUndObj = getUnderlyingObject(KillingLoc.Ptr);
    LLVM_DEBUG(dbgs() << "Trying to eliminate MemoryDefs killed by "
                      << *KillingDef << " (" << *KillingI << ")\n");

    unsigned ScanLimit = MemorySSAScanLimit;
    unsigned WalkerStepLimit = MemorySSAUpwardsStepLimit;
    unsigned PartialLimit = MemorySSAPartialStoreLimit;
    // Worklist of MemoryAccesses that may be killed by KillingDef.
    SetVector<MemoryAccess *> ToCheck;
    ToCheck.insert(KillingDef->getDefiningAccess());

    bool Shortend = false;
    bool IsMemTerm = State.isMemTerminatorInst(KillingI);
    // Check if MemoryAccesses in the worklist are killed by KillingDef.
    for (unsigned I = 0; I < ToCheck.size(); I++) {
      MemoryAccess *Current = ToCheck[I];
      if (State.SkipStores.count(Current))
        continue;

      std::optional<MemoryAccess *> MaybeDeadAccess = State.getDomMemoryDef(
          KillingDef, Current, KillingLoc, KillingUndObj, ScanLimit,
          WalkerStepLimit, IsMemTerm, PartialLimit);

      if (!MaybeDeadAccess) {
        LLVM_DEBUG(dbgs() << "  finished walk\n");
        continue;
      }

      MemoryAccess *DeadAccess = *MaybeDeadAccess;
      LLVM_DEBUG(dbgs() << " Checking if we can kill " << *DeadAccess);
      if (isa<MemoryPhi>(DeadAccess)) {
        LLVM_DEBUG(dbgs() << "\n  ... adding incoming values to worklist\n");
        for (Value *V : cast<MemoryPhi>(DeadAccess)->incoming_values()) {
          MemoryAccess *IncomingAccess = cast<MemoryAccess>(V);
          BasicBlock *IncomingBlock = IncomingAccess->getBlock();
          BasicBlock *PhiBlock = DeadAccess->getBlock();

          // We only consider incoming MemoryAccesses that come before the
          // MemoryPhi. Otherwise we could discover candidates that do not
          // strictly dominate our starting def.
          if (State.PostOrderNumbers[IncomingBlock] >
              State.PostOrderNumbers[PhiBlock])
            ToCheck.insert(IncomingAccess);
        }
        continue;
      }
      auto *DeadDefAccess = cast<MemoryDef>(DeadAccess);
      Instruction *DeadI = DeadDefAccess->getMemoryInst();
      LLVM_DEBUG(dbgs() << " (" << *DeadI << ")\n");
      ToCheck.insert(DeadDefAccess->getDefiningAccess());
      NumGetDomMemoryDefPassed++;

      if (!DebugCounter::shouldExecute(MemorySSACounter))
        continue;

      MemoryLocation DeadLoc = *State.getLocForWrite(DeadI);

      if (IsMemTerm) {
        const Value *DeadUndObj = getUnderlyingObject(DeadLoc.Ptr);
        if (KillingUndObj != DeadUndObj)
          continue;
        LLVM_DEBUG(dbgs() << "DSE: Remove Dead Store:\n  DEAD: " << *DeadI
                          << "\n  KILLER: " << *KillingI << '\n');
        State.deleteDeadInstruction(DeadI);
        LLVM_DEBUG(dbgs() << "State.deleteDeadInstruction(DeadI);\n");
        ++NumFastStores;
        MadeChange = true;
      } else {
        // Check if DeadI overwrites KillingI.
        int64_t KillingOffset = 0;
        int64_t DeadOffset = 0;
        OverwriteResult OR = State.isOverwrite(
            KillingI, DeadI, KillingLoc, DeadLoc, KillingOffset, DeadOffset);
        if (OR == OW_MaybePartial) {
          auto Iter = State.IOLs.insert(
              std::make_pair<BasicBlock *, InstOverlapIntervalsTy>(
                  DeadI->getParent(), InstOverlapIntervalsTy()));
          auto &IOL = Iter.first->second;
          OR = isPartialOverwrite(KillingLoc, DeadLoc, KillingOffset,
                                  DeadOffset, DeadI, IOL);
        }

        if (EnablePartialStoreMerging && OR == OW_PartialEarlierWithFullLater) {
          auto *DeadSI = dyn_cast<StoreInst>(DeadI);
          auto *KillingSI = dyn_cast<StoreInst>(KillingI);
          // We are re-using tryToMergePartialOverlappingStores, which requires
          // DeadSI to dominate DeadSI.
          // TODO: implement tryToMergeParialOverlappingStores using MemorySSA.
          if (DeadSI && KillingSI && DT.dominates(DeadSI, KillingSI)) {
            if (Constant *Merged = tryToMergePartialOverlappingStores(
                    KillingSI, DeadSI, KillingOffset, DeadOffset, State.DL,
                    State.BatchAA, &DT)) {

              // Update stored value of earlier store to merged constant.
              DeadSI->setOperand(0, Merged);
              LLVM_DEBUG(dbgs() << "DeadSI->setOperand(0, Merged);\n");
              ++NumModifiedStores;
              MadeChange = true;

              Shortend = true;
              // Remove killing store and remove any outstanding overlap
              // intervals for the updated store.
              State.deleteDeadInstruction(KillingSI);
              auto I = State.IOLs.find(DeadSI->getParent());
              if (I != State.IOLs.end())
                I->second.erase(DeadSI);
              break;
            }
          }
        }

        if (OR == OW_Complete) {
          LLVM_DEBUG(dbgs() << "DSE: Remove Dead Store:\n  DEAD: " << *DeadI
                            << "\n  KILLER: " << *KillingI << '\n');
          State.deleteDeadInstruction(DeadI);
          ++NumFastStores;
          MadeChange = true;
        }
      }
    }

    // Check if the store is a no-op.
    if (!Shortend && State.storeIsNoop(KillingDef, KillingUndObj)) {
      LLVM_DEBUG(dbgs() << "DSE: Remove No-Op Store:\n  DEAD: " << *KillingI
                        << '\n');
      State.deleteDeadInstruction(KillingI);
      LLVM_DEBUG(dbgs() << "State.deleteDeadInstruction(KillingI);\n");
      NumRedundantStores++;
      MadeChange = true;
      continue;
    }

    // Can we form a calloc from a memset/malloc pair?
    if (!Shortend && State.tryFoldIntoCalloc(KillingDef, KillingUndObj)) {
      LLVM_DEBUG(dbgs() << "DSE: Remove memset after forming calloc:\n"
                        << "  DEAD: " << *KillingI << '\n');
      State.deleteDeadInstruction(KillingI);
      MadeChange = true;
      continue;
    }
  }

  MadeChange |= State.eliminateRedundantStoresOfExistingValues();
  MadeChange |= State.eliminateDeadWritesAtEndOfFunction();
  return MadeChange;
}

