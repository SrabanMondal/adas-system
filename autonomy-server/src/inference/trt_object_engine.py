import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class TRTObjectInferenceEngine:
    """
    Clean, production-safe TensorRT inference engine.
    Fully compatible with OpenVINO output format.
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(idx):
                self.input_shape = shape  # (1, 3, 320, 320)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_shape = shape  # e.g. (1, 84, 2100)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess for TensorRT engine.
        """
        _, _, H, W = self.input_shape
        img = cv2.resize(frame, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, 0))

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns (1, 300, 6) detections exactly like OpenVINO.
        """
        img = self._preprocess(frame)
        np.copyto(self.host_inputs[0], img.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        # Expect engine output shape = (1, 84, 2100)
        arr = self.host_outputs[0].reshape(self.output_shape)
        return self._decode_v5_to_v26(arr) # (300,6)
    
    def _decode_v5_to_v26(self, raw_output, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        Transforms YOLOv5 1x84x2100 raw output to YOLOv26-style 1x300x6.
        Shape: [N, 6] -> [x1, y1, x2, y2, conf, class_id]
        """
        # 1. Reshape and Filter (conf = obj_conf * class_conf)
        # raw_output is (84, 2100) -> Transpose to (2100, 84)
        x = raw_output.T
        
        # YOLOv5: [cx, cy, w, h, obj_conf, cls0...cls79]
        obj_conf = x[:, 4]
        cls_confs = x[:, 5:]
        
        # Calculate combined confidence
        best_cls = np.argmax(cls_confs, axis=1)
        best_cls_conf = cls_confs[np.arange(len(x)), best_cls]
        combined_conf = obj_conf * best_cls_conf
        
        # Filter by threshold
        mask = combined_conf > conf_thres
        x = x[mask]
        combined_conf = combined_conf[mask]
        best_cls = best_cls[mask]
        
        if len(x) == 0:
            return np.zeros((max_det, 6))

        # 2. Convert cx, cy, w, h -> x1, y1, x2, y2
        # Input size is 128
        boxes = np.zeros((len(x), 4))
        boxes[:, 0] = x[:, 0] - x[:, 2] / 2 # x1
        boxes[:, 1] = x[:, 1] - x[:, 3] / 2 # y1
        boxes[:, 2] = x[:, 0] + x[:, 2] / 2 # x2
        boxes[:, 3] = x[:, 1] + x[:, 3] / 2 # y2

        # 3. Fast Vectorized NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            combined_conf.tolist(), 
            conf_thres, 
            iou_thres
        )

        # 4. Final Formatting to 300x6
        final_output = np.zeros((max_det, 6))
        if len(indices) > 0:
            # Flatten indices if needed
            idx = np.array(indices).flatten()[:max_det]
            count = len(idx)
            
            final_output[:count, :4] = boxes[idx]
            final_output[:count, 4] = combined_conf[idx]
            final_output[:count, 5] = best_cls[idx]

        return final_output