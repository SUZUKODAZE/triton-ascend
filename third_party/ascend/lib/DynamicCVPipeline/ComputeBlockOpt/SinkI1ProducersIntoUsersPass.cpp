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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "sink-i1-producers-into-users";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;

namespace mlir {
namespace triton {

class SinkI1ProducersIntoUsersPass
    : public PassWrapper<SinkI1ProducersIntoUsersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkI1ProducersIntoUsersPass)

  SinkI1ProducersIntoUsersPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "sink-i1-producers-into-users";
  }
  llvm::StringRef getDescription() const final {
    return "Sink i1-producing ops next to each of their i1 uses";
  }
};

} // namespace triton
} // namespace mlir

namespace {

static bool isI1Producer(Operation *op) {
  for (auto result : op->getResults()) {
    if (auto tensorType = dyn_cast<mlir::TensorType>(result.getType())) {
      if (tensorType.getElementType().isInteger(1))
        return true;
    }
  }
  return false;
}

static bool isPureAndRegionless(Operation *op) {
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    if (!effects.empty())
      return false;
  }
  return true;
}

} // namespace

namespace mlir {
namespace triton {

void SinkI1ProducersIntoUsersPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  LOG_DEBUG("=== SinkI1ProducersIntoUsersPass start ===\n");
  LOG_DEBUG("Module: " << moduleOp.getName() << "\n");

  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(moduleOp, aa);
  CVPipeline::ComputeBlockIdManager bm(moduleOp);

  SmallVector<Operation *> producers;
  moduleOp.walk([&](Operation *op) {
    if (isI1Producer(op) && isPureAndRegionless(op)) {
      LOG_DEBUG("found i1 producer: " << op->getName() << "\n");
      producers.push_back(op);
    }
  });
  LOG_DEBUG("total i1 producers found: " << producers.size() << "\n");

  for (Operation *p : producers) {
    LOG_DEBUG("processing producer: " << p->getName() << "\n");
    bool hasSameBlockUser = false;

    for (OpOperand &use : p->getUses()) {
      Type t = use.get().getType();
      if (auto tensorType = dyn_cast<mlir::TensorType>(t)) {
        mlir::Type elemType = tensorType.getElementType();
        if (elemType.isInteger(1)) {
          Operation *consumer = use.getOwner();
          LOG_DEBUG("  consumer: " << consumer->getName() << "\n");
          if (bm.isSameBlock(p, consumer)) {
            LOG_DEBUG("    same block user, skip\n");
            hasSameBlockUser = true;
            continue;
          }

          unsigned idx = 0;
          for (unsigned i = 0; i < p->getNumResults(); ++i) {
            if (p->getResult(i) == use.get()) {
              idx = i;
              break;
            }
          }

          int consumerBlockId = bm.getBlockIdByOp(consumer);
          LOG_DEBUG("    consumerBlockId: " << consumerBlockId << "\n");
          SmallVector<Operation *> opsToCheck = {p};
          if (CVPipeline::willCreateCycle(opsToCheck, memGraph, consumerBlockId, bm)) {
            LOG_DEBUG("    willCreateCycle=true, skip\n");
            continue;
          }
          LOG_DEBUG("    willCreateCycle=false, cloning producer\n");

          Operation *cloned = p->clone();
          consumer->getBlock()->push_back(cloned);
          cloned->moveBefore(consumer);
          LOG_DEBUG("    cloned: " << cloned->getName() << "\n");
          if (consumerBlockId != -1) {
            cloned->setAttr(mlir::CVPipeline::kBlockId,
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(p->getContext(), 32),
                                consumerBlockId));
          }
          consumer->getOpOperand(use.getOperandNumber()).set(cloned->getResult(idx));
        }
      }
    }

    if (!hasSameBlockUser && p->use_empty()) {
      LOG_DEBUG("  erasing producer (no same-block user, use_empty=true): " << p->getName() << "\n");
      p->erase();
    } else if (!hasSameBlockUser) {
      LOG_DEBUG("  keep producer (has users but no same-block user): " << p->getName() << "\n");
    }
  }
  LOG_DEBUG("=== SinkI1ProducersIntoUsersPass done ===\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersIntoUsersPass() {
  return std::make_unique<SinkI1ProducersIntoUsersPass>();
}

} // namespace triton
} // namespace mlir