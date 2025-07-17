# models/superpoint.py

from models.superglue_author.models.superpoint import SuperPoint as _OfficialSP


def SuperPoint(config):
    """
    config: dict from feature.superpoint in your YAML.
    Passed straight to the official SuperPoint __init__.
    """
    return _OfficialSP(config)
