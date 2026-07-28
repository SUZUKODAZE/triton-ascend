/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iter-var-opt"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)
#define PRINT_DEBUG(msg) llvm::errs() << " [iter-var-opt-DEBUG] " << msg << "\n" << flush

using namespace mlir;
using namespace triton;
using namespace mlir::triton;

namespace mlir {
namespace triton {

class IterVarOptPass : public PassWrapper<IterVarOptPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterVarOptPass)

  IterVarOptPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "iter-var-opt"; }

private:
struct IterArgsInfo {
    BlockArgument iterArgs;
    Operation *updateOp = nullptr;
    llvm::DenseSet<Operation *> usageOps;
    int updateBlockId = -1;
  };

  void collectIterArgs(Block *block, llvm::DenseMap<Operation *, IterArgsInfo> &iterArgsInfos,
                       llvm::DenseMap<int, llvm::SmallVector<Operation *>> &blockIdToUpdateOps,
                       CVPipeline::ComputeBlockIdManager &bm);

  bool splitAndMergeUsageOps(IterArgsInfo &info, Block *loopBody,
                             const CVPipeline::MemoryDependenceGraph &memGraph,
                             CVPipeline::ComputeBlockIdManager &bm);

  void processLoopBlock(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
                        CVPipeline::ComputeBlockIdManager &bm);
};

void IterVarOptPass::collectIterArgs(Block *block, llvm::DenseMap<Operation *, IterArgsInfo> &iterArgsInfos,
                                     llvm::DenseMap<int, llvm::SmallVector<Operation *>> &blockIdToUpdateOps,
                                     CVPipeline::ComputeBlockIdManager &bm) {
  auto yieldOp = dyn_cast<scf::YieldOp>(block->getTerminator());
  unsigned numArgs = block->getNumArguments();
  unsigned numYieldOperands = yieldOp.getNumOperands();
  int offset = (int)numArgs - (int)numYieldOperands;

  for (unsigned argIdx = 0; argIdx < numYieldOperands; ++argIdx) {
    int iterArgIdx = argIdx + offset;
    if (iterArgIdx < 0 || iterArgIdx >= (int)numArgs)
      continue;

    BlockArgument iterArg = block->getArgument(iterArgIdx);
    if (!isa<RankedTensorType>(iterArg.getType()))
      continue;

    IterArgsInfo info;
    info.iterArgs = iterArg;
    info.updateBlockId = -1;

    Value yieldedVal = yieldOp.getOperand(argIdx);
    if (Operation *defOp = yieldedVal.getDefiningOp()) {
      LOG_DEBUG("collecting iterArg: " << iterArg << " with updateOp: " << *defOp << "\n");
      Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
      if (defInBlock && defInBlock->getBlock() == block) {
        info.updateOp = defInBlock;
        info.updateBlockId = bm.getBlockIdByOp(defInBlock);
        blockIdToUpdateOps[info.updateBlockId].push_back(defInBlock);
      }
      for (Operation *user : iterArg.getUsers()) {
        Operation *userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (userInBlock && userInBlock->getBlock() == block && bm.getBlockIdByOp(userInBlock) != info.updateBlockId) {
          info.usageOps.insert(userInBlock);
        }
      }
      if (info.updateOp && !info.usageOps.empty()) {
        iterArgsInfos[info.updateOp] = info;
      }
    }
    
  }
}

bool IterVarOptPass::splitAndMergeUsageOps(IterArgsInfo &info, Block *loopBody,
                                           const CVPipeline::MemoryDependenceGraph &memGraph,
                                           CVPipeline::ComputeBlockIdManager &bm) {
  int updateBlockId = info.updateBlockId;

  PRINT_DEBUG("===== splitAndMergeUsageOps START =====");
  PRINT_DEBUG("updateOp: " << info.updateOp->getName() << " at blockId=" << updateBlockId);
  PRINT_DEBUG("iterArgs: " << info.iterArgs << " (arg" << info.iterArgs.getArgNumber() << ")");
  PRINT_DEBUG("updateOp->getUsers() count: " << info.updateOp->getUsers().size());
  PRINT_DEBUG("usageOps count: " << info.usageOps.size());
  
  for (Operation *op : info.usageOps) {
    int opBlockId = bm.getBlockIdByOp(op);
    PRINT_DEBUG("  usageOp: " << op->getName() << " at blockId=" << opBlockId);
    PRINT_DEBUG("    usageOp->getUsers() count: " << op->getUsers().size());
    for (Operation *user : op->getUsers()) {
      Operation *userInBlock = CVPipeline::getAncestorInBlock(user, loopBody);
      if (userInBlock) {
        int userBlockId = bm.getBlockIdByOp(userInBlock);
        PRINT_DEBUG("      -> user: " << userInBlock->getName() << " at blockId=" << userBlockId);
      }
    }
  }

  llvm::DenseSet<Operation *> visited;
  llvm::SmallVector<Operation *> allOpsToMove;

  std::function<void(Operation *, int)> collectOpsToMove = [&](Operation *op, int originalBlockId) {
    if (visited.count(op))
      return;
    visited.insert(op);
    int opBlockId = bm.getBlockIdByOp(op);
    if (opBlockId != originalBlockId)
      return;
    allOpsToMove.push_back(op);
    PRINT_DEBUG("  [collectOpsToMove] op: " << op->getName() << " at blockId=" << opBlockId);

    for (Operation *user : op->getUsers()) {
      Operation *userInBlock = CVPipeline::getAncestorInBlock(user, loopBody);
      if (userInBlock) {
        collectOpsToMove(userInBlock, originalBlockId);
      }
    }
  };

  llvm::SmallVector<int> usageBlockIds;
  for (Operation *usageOp : info.usageOps) {
    int usageBlockId = bm.getBlockIdByOp(usageOp);
    usageBlockIds.push_back(usageBlockId);
    visited.clear();
    PRINT_DEBUG("  Starting collection from usageOp: " << usageOp->getName() << " (blockId=" << usageBlockId << ")");
    collectOpsToMove(usageOp, usageBlockId);
  }

  PRINT_DEBUG("  allOpsToMove collected: " << allOpsToMove.size());
  for (auto op : allOpsToMove) {
    int opBlockId = bm.getBlockIdByOp(op);
    PRINT_DEBUG("    op in allOpsToMove: " << op->getName() << " at blockId=" << opBlockId);
  }

  auto finalOneBlockId = bm.getNextId();
  PRINT_DEBUG("  Calling willCreateCycle with " << allOpsToMove.size() << " ops, targetBlockId=" << finalOneBlockId);
  
  bool willCreateCycleResult = CVPipeline::willCreateCycle(allOpsToMove, memGraph, finalOneBlockId, bm);
  
  if (willCreateCycleResult) {
    PRINT_DEBUG("  *** willCreateCycle returned TRUE - CYCLE DETECTED! ***");
    PRINT_DEBUG("===== splitAndMergeUsageOps END (return false - cycle detected) =====\n");
    return false;
  } else {
    PRINT_DEBUG("  ==== willCreateCycle returned FALSE - NO CYCLE ====");
  }

  llvm::DenseMap<Operation *, int> opToNewBlockId;
  int currentNewBlockId = bm.getNextId();
  for (size_t i = 0; i < usageBlockIds.size(); ++i) {
    int originalBlockId = usageBlockIds[i];
    for (Operation *op : allOpsToMove) {
      if (bm.getBlockIdByOp(op) == originalBlockId) {
        opToNewBlockId[op] = currentNewBlockId;
        PRINT_DEBUG("  [assign] op " << op->getName() << " -> blockId " << currentNewBlockId);
      }
    }
    currentNewBlockId = bm.getNextId();
  }

  PRINT_DEBUG("  Applying blockId changes...");
  for (auto &kv : opToNewBlockId) {
    bm.updateBlockId(kv.first, kv.second);
    PRINT_DEBUG("    Changed op " << kv.first->getName() << " to blockId " << kv.second);
  }

  PRINT_DEBUG("  Merging all to finalOneBlockId=" << finalOneBlockId);
  for (auto &kv : opToNewBlockId) {
    bm.updateBlockId(kv.first, finalOneBlockId);
    PRINT_DEBUG("    Merged op " << kv.first->getName() << " to finalOneBlockId " << finalOneBlockId);
  }

  PRINT_DEBUG("===== splitAndMergeUsageOps END (return true) =====\n");
  return true;
}

void IterVarOptPass::processLoopBlock(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
                                      CVPipeline::ComputeBlockIdManager &bm) {
  llvm::DenseMap<Operation *, IterArgsInfo> iterArgsInfos;
  llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToUpdateOps;

  collectIterArgs(block, iterArgsInfos, blockIdToUpdateOps, bm);

  if (iterArgsInfos.empty())
    return;

  LOG_DEBUG("Collected " << iterArgsInfos.size() << " iteration variables in block\n");

  for (auto &entry : iterArgsInfos) {
    LOG_DEBUG("Processing iteration variable with update op: " << *entry.first << "\n");
    bool result = splitAndMergeUsageOps(entry.second, block, memGraph, bm);
    PRINT_DEBUG("[processLoopBlock] splitAndMergeUsageOps returned: " << result);
  }
}

void IterVarOptPass::runOnOperation() {
  LOG_DEBUG("--- Pass: IterVarOpt ---\n");

  ModuleOp module = getOperation();
  LOG_DEBUG(" input "<< module << "\n");

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);
  CVPipeline::ComputeBlockIdManager bm(module);

  llvm::SmallVector<Block *> blocks;
  module.walk([&](Block *block) { blocks.push_back(block); });

  for (Block *block : blocks) {
    if (!block || block->empty())
      continue;
    if(!block->mightHaveTerminator())
      continue;
    auto *terminator = block->getTerminator();
    if (!terminator || !isa<scf::YieldOp>(terminator))
      continue;

    processLoopBlock(block, memDepGraph, bm);
  }

  LOG_DEBUG("=== Pass IterVarOpt complete ===\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createIterVarOptPass() {
  return std::make_unique<IterVarOptPass>();
}

} // namespace triton
} // namespace mlir
