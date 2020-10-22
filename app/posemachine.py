import sys

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine')
from keypoints_extractor import KeypointsExtractor
from detectors_keras_api.initial_pose_detector_keras import InitialPoseDetector
from detectors_keras_api.right_hand_detector_keras import RightHandUpDetector


class PoseMachine():
    __instance = None

    @staticmethod
    def get_instance():
        if PoseMachine.__instance == None:
            PoseMachine()
        return PoseMachine.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if PoseMachine.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.target_id = ""
            self.exercise_type = ""
            self.pose_predictions = []
            self.kp_extractor = KeypointsExtractor()
            self.init_pose_detector = InitialPoseDetector()
            self.right_hand_up_detector = RightHandUpDetector()
            PoseMachine.__instance = self
    
    @staticmethod
    def dispose():
        PoseMachine.__instance = None


if __name__ == "__main__":
    pm1 = PoseMachine.get_instance()
    pm2 = PoseMachine.get_instance()

    print("pm1 kp_extractor -->", pm1.kp_extractor)
    print("pm2 kp_extractor -->", pm2.kp_extractor)