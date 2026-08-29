"""SIFT + affine-cluster copy-move forgery localization."""

from collections import defaultdict
from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


MIN_KEYPOINTS = 100
MIN_OFFSET = 32.0
GRID_SIZE = 8.0


def _cluster_matches(keypoints: list[cv2.KeyPoint], descriptors: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    clusters: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for candidates in matcher.knnMatch(descriptors, descriptors, k=3):
        others = [match for match in candidates if match.trainIdx != match.queryIdx]
        if len(others) < 2:
            continue
        best, second = others[:2]
        if second.distance <= 0 or best.distance >= 0.75 * second.distance:
            continue
        source = np.asarray(keypoints[best.queryIdx].pt, dtype=np.float32)
        destination = np.asarray(keypoints[best.trainIdx].pt, dtype=np.float32)
        offset = destination - source
        if float(np.linalg.norm(offset)) < MIN_OFFSET:
            continue
        key = tuple(np.rint(offset / GRID_SIZE).astype(int))
        clusters[key].append((best.queryIdx, best.trainIdx))
    return {key: matches for key, matches in clusters.items() if len(matches) >= 8}


def _visualization(image: np.ndarray, hulls: list[np.ndarray], arrows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    output = image.copy()
    for hull in hulls:
        cv2.polylines(output, [hull.astype(np.int32)], True, (0, 255, 0), 3)
    for source, destination in arrows:
        source_point = tuple(np.mean(source.reshape(-1, 2), axis=0).astype(int))
        destination_point = tuple(np.mean(destination.reshape(-1, 2), axis=0).astype(int))
        cv2.arrowedLine(output, source_point, destination_point, (255, 64, 0), 3, tipLength=0.15)
    return output


class CopyMoveDetector:
    id = "copy_move"
    name = "Copy-Move Forgery Localization"
    family = "geometric"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Finds duplicated image regions using SIFT descriptors and affine RANSAC clusters."
    limitations = ["Textureless pasted regions produce insufficient keypoints and are not a clean verdict."]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format and ctx.format not in self.applicable_formats:
            return False, f"copy-move does not support decoded format {ctx.format}"
        return True, "copy-move matching supports this image format"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(self.id, DetectorState.NOT_APPLICABLE, None, None, float(config["threshold"]), reason, {}, None, _duration(started))

        image = ctx.downscaled_rgb_uint8
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=5000)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        base_metrics = {"keypoints": float(len(keypoints)), "verified_clusters": 0.0, "inlier_count": 0.0}
        if descriptors is None or len(keypoints) < MIN_KEYPOINTS:
            mask = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
            return DetectorResult(
                self.id, DetectorState.NOT_APPLICABLE, None, None, float(config["threshold"]),
                "insufficient texture for keypoint matching: fewer than 100 keypoints", base_metrics,
                cv2.resize(mask, (ctx.width, ctx.height), interpolation=cv2.INTER_NEAREST), _duration(started),
            )

        verified: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for _, matches in sorted(_cluster_matches(keypoints, descriptors).items(), key=lambda item: len(item[1]), reverse=True):
            source = np.float32([keypoints[query].pt for query, _ in matches])
            destination = np.float32([keypoints[train].pt for _, train in matches])
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if affine is None or inlier_mask is None or int(inlier_mask.sum()) < 8:
                continue
            inliers = inlier_mask.ravel().astype(bool)
            verified.append((int(inliers.sum()), source[inliers], destination[inliers], affine))

        overlay = image.copy()
        if not verified:
            score = to_probability(0.0, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
            return DetectorResult(
                self.id, DetectorState.APPLICABLE, score, score >= 0.5, float(config["threshold"]),
                "no_forgery_found: keypoints were sufficient but no verified affine cluster was found",
                base_metrics, cv2.resize(overlay, (ctx.width, ctx.height), interpolation=cv2.INTER_LINEAR), _duration(started),
            )

        hulls: list[np.ndarray] = []
        arrows: list[tuple[np.ndarray, np.ndarray]] = []
        for _, source, destination, _ in verified:
            hulls.extend((cv2.convexHull(source), cv2.convexHull(destination)))
            arrows.append((source, destination))
        overlay = _visualization(image, hulls, arrows)
        scale_x = ctx.width / image.shape[1]
        scale_y = ctx.height / image.shape[0]
        best = verified[0]
        best_source, best_destination = best[1], best[2]
        translation = best[3][:, 2]
        metrics = {
            "keypoints": float(len(keypoints)),
            "verified_clusters": float(len(verified)),
            "inlier_count": float(best[0]),
            "translation_dx": float(translation[0] * scale_x),
            "translation_dy": float(translation[1] * scale_y),
            "affine_scale": float(np.linalg.norm(best[3][0, :2])),
            "affine_rotation_degrees": float(np.degrees(np.arctan2(best[3][1, 0], best[3][0, 0]))),
            "mask_fraction": float(sum(cv2.contourArea(hull) for hull in hulls) / max(1, ctx.width * ctx.height)),
        }
        raw = float(len(verified))
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, score >= 0.5, float(config["threshold"]),
            f"verified {len(verified)} affine copy-move cluster(s); best cluster has {best[0]} inliers",
            metrics, cv2.resize(overlay, (ctx.width, ctx.height), interpolation=cv2.INTER_LINEAR), _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
