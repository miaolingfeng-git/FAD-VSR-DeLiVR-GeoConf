from .video_sr import (PairedVideoDataset, VideoPairFolder, VideoSRDataset,
                        VimeoSeptupletDataset, VimeoSeptupletFolder)


def make(spec):
    name = spec.get("name", "paired-video")
    args = spec.get("args", {})
    if name in {"paired-video", "PairedVideoDataset", "video-sr"}:
        return PairedVideoDataset(**args)
    if name in {"video-folder", "VideoPairFolder"}:
        return VideoPairFolder(**args)
    if name in {"vimeo-septuplet", "VimeoSeptupletDataset"}:
        return VimeoSeptupletDataset(**args)
    if name in {"vimeo-folder", "VimeoSeptupletFolder"}:
        return VimeoSeptupletFolder(**args)
    raise ValueError(f"unknown dataset: {name}")

