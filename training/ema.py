import copy
import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self):
        for name, param in self.model.state_dict().items():
            self.shadow[name] = (
                self.decay * self.shadow[name]
                + (1.0 - self.decay) * param
            )

    def apply_shadow(self):
        self.model.load_state_dict(self.shadow)
