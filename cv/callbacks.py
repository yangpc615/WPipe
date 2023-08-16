
import torch
from torch.utils.data._utils.collate import default_collate


def resnext_callbacks(self, x, labels):
    output = {}
    if hasattr(self, "conv1"):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

    if hasattr(self, "layer") and self.layer is not None:
        x = self.layer(x)

    if hasattr(self, "fc"):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        loss = self.loss_func(x, labels)
        output["loss"] = loss
        output["output"] = x
        output["labels"] = labels
        return output

    return {"x": x, "labels": labels}


def resnext_collate(batch):
    output = default_collate(batch)
    input_keys = ("x", "labels")
    output = {k: v for k, v in zip(input_keys, output)}
    return output
