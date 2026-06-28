from control_models.base import ControlModel


class ChoicesModel(ControlModel):
    @classmethod
    def is_control_matched(cls, control):
        return False

    @classmethod
    def create(cls, yolo_instance, control):
        return None
