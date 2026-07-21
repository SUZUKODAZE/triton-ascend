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

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/SinkI1Producers.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Returns true if `op` produces a tensor whose element type is i1
// (e.g. tensor<...xi8> used as a boolean mask).
static bool isI1Producer(Operation *op)
{
    for (auto result : op->getResults()) {
        if (auto tensorType = dyn_cast<mlir::TensorType>(result.getType())) {
            mlir::Type elemType = tensorType.getElementType();
            if (elemType.isInteger(1)) {
                return true;
            }
        }
    }
    return false;
}

// Returns true if `op` is safe to duplicate: no recursive memory effects
// and no memory effect interface. arith.cmpi/cmpf/andi/ori/xori/trunci and
// similar pure value ops fall in this set; scf.if, func.func, loads/stores,
// and any op with memory effects are excluded.
static bool isPureAndRegionless(Operation *op)
{
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

namespace mlir::triton {

void SinkI1ProducersPass::runOnOperation()
{
    ModuleOp moduleOp = getOperation();

    // Build a block-id view of the module. This pass is expected to run
    // after PlanCubeBlock / PlanVectorBlock have assigned ssbuffer.block_id.
    // Strategy:
    //   - Same-block i1 consumers: leave alone (producer is already
    //     local; no cross-block transport of the i1 value).
    //   - Cross-block i1 consumers: clone the producer right before the
    //     consumer and reassign the clone's ssbuffer.block_id to the
    //     consumer's block. The i1 value is then computed locally in the
    //     consumer's block, avoiding the cross-block transport of the
    //     i1 result. The original producer is kept around for any
    //     remaining same-block users.
    //   - If no same-block i1 user remains after rewiring, drop the
    //     original producer.
    CVPipeline::ComputeBlockIdManager bm(moduleOp);

    SmallVector<Operation *> producers;
    moduleOp.walk([&](Operation *op) {
        llvm::errs() << "[DEBUG] checking op: " << op->getName() << "\n";
        if (isI1Producer(op)) {
            llvm::errs() << "  -> is i1 producer\n";
            llvm::errs() << "  -> regions: " << op->getNumRegions() << "\n";
            llvm::errs() << "  -> HasRecursiveMemoryEffects: " << op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() << "\n";
            if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
                SmallVector<MemoryEffects::EffectInstance> effects;
                iface.getEffects(effects);
                llvm::errs() << "  -> MemoryEffectOpInterface effects: " << effects.size() << "\n";
            } else {
                llvm::errs() << "  -> MemoryEffectOpInterface: not implemented\n";
            }
            if (!isPureAndRegionless(op)) {
                llvm::errs() << "  -> SKIP: not pure\n";
                return;
            }
            llvm::errs() << "  -> ACCEPTED\n";
            producers.push_back(op);
        }
    });

    llvm::errs() << "[SinkI1Producers] found " << producers.size() << " producers\n";

    for (Operation *p : producers) {
        llvm::errs() << "[SinkI1Producers] processing producer: " << p->getName() << " at block_id " << bm.getBlockIdByOp(p) << "\n";
        SmallVector<std::pair<OpOperand *, unsigned>> i1Uses;
        for (OpOperand &use : p->getUses()) {
            Type t = use.get().getType();
            bool isi1 = false;
            if (auto tensorType = dyn_cast<mlir::TensorType>(t)) {
                mlir::Type elemType = tensorType.getElementType();
                if (elemType.isInteger(1)) {
                    isi1 = true;
                    llvm::errs() << "  -> use in tensor<...xi1>: " << use.getOperandNumber() << "\n";
                }
            }
            if (isi1) {
                unsigned idx = use.getOperandNumber();
                i1Uses.push_back({&use, idx});
            }
        }

        llvm::errs() << "  -> i1Uses count: " << i1Uses.size() << "\n";

        if (i1Uses.empty()) {
            p->erase();
            continue;
        }

        bool hasSameBlockUser = false;

        for (auto [use, resultIdx] : i1Uses) {
            Operation *consumer = use->getOwner();
            int producerBlockId = bm.getBlockIdByOp(p);
            int consumerBlockId = bm.getBlockIdByOp(consumer);
            llvm::errs() << "  -> consumer: " << consumer->getName() << " producer_block=" << producerBlockId << " consumer_block=" << consumerBlockId << "\n";

            if (bm.isSameBlock(p, consumer)) {
                llvm::errs() << "     SAME block, skip\n";
                hasSameBlockUser = true;
                continue;
            }

            llvm::errs() << "     CROSS block, clone and insert\n";
            Operation *cloned = p->clone();
            consumer->getBlock()->push_back(cloned);
            cloned->moveBefore(consumer);
            if (consumerBlockId != -1) {
                cloned->setAttr(mlir::CVPipeline::kBlockId,
                                mlir::IntegerAttr::get(
                                    mlir::IntegerType::get(p->getContext(),
                                                           /*bitwidth=*/32),
                                    consumerBlockId));
            }
            use->set(cloned->getResult(resultIdx));
        }

        if (!hasSameBlockUser && p->use_empty()) {
            llvm::errs() << "  -> erase original (no same-block user)\n";
            p->erase();
        }
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersPass()
{
    return std::make_unique<SinkI1ProducersPass>();
}

void registerSinkI1ProducersPasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createSinkI1ProducersPass();
    });
}

} // namespace mlir::triton