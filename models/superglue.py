# models/superglue.py

from models.superglue_author.models.superglue import SuperGlue as _OfficialSG


def SuperGlue(config):
    """
    config: the dict under matcher.superglue in your YAML
            (keys: weights, sinkhorn_iterations, match_threshold, device, etc.)
    Returns the official SuperGlue model loaded with those settings.
    """
    return _OfficialSG(config)
