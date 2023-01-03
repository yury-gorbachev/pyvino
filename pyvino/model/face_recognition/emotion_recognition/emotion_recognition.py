import numpy as np

from ...openvino_model.basic_model import BasicModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class EmotionRecognition(BasicModel):
    model_name = 'emotions-recognition-retail-0003'
    model_loc = 'intel'
    label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)
    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        result = {}
        irequest=self.model_requests[cur_request_id]
        irequest.wait()

        probs=irequest.get_tensor('prob_emotion').data.flatten()

        emotion_id = np.argmax(probs)
        emotion = self.label[emotion_id]
        emotion_prob = probs[emotion_id]
            
        result['emotion'] = emotion
        result['emotion_prob'] = emotion_prob
        
        return result
