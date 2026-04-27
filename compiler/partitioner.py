import onnx
import numpy as np

class HyPEPartitioner:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph

    def partition(self):
        """Splits graph into MPC (Linear) and TEE (Non-Linear) sets."""
        mpc_nodes = []
        tee_nodes = []
        
        non_linear_ops = ['Relu', 'MaxPool', 'BatchNorm', 'Sigmoid']
        
        for node in self.graph.node:
            if node.op_type in non_linear_ops:
                tee_nodes.append(node.name)
            else:
                mpc_nodes.append(node.name)
        
        return {"MPC": mpc_nodes, "TEE": tee_nodes}

if __name__ == "__main__":
    print("HyPE Partitioner initialized.")
