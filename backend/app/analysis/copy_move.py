"""SIFT + affine-cluster copy-move forgery localization."""

from collections import defaultdict
from time import perf_counter

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


MIN_KEYPOINTS = 100
GENERALIZED_RATIO_THRESHOLD = 0.5
CLUSTER_INCONSISTENCY_THRESHOLD = 2.2
CLUSTER_DEPTH = 4
MIN_CLUSTER_MATCHES = 3
MATCH_BATCH_SIZE = 64
MIN_AFFINE_SCALE = 0.5
MAX_AFFINE_SCALE = 2.5
MAX_AFFINE_CONDITION = 3.0
# ponytail: fixed 8-inlier confidence floor suppresses repeated glyph/texture matches; dense
# copy-move scenes need a separately calibrated confidence model if this becomes a required class.
MIN_CONFIDENT_INLIERS = 8


def _generalized_matches(descriptors: np.ndarray) -> list[tuple[int, int]]:
    """Return Amerini's generalized 2NN matches, excluding self-matches."""
    count = len(descriptors)
    if count < 3:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches: list[tuple[int, int]] = []
    for start in range(0, count, MATCH_BATCH_SIZE):
        stop = min(count, start + MATCH_BATCH_SIZE)
        rows = matcher.knnMatch(descriptors[start:stop], descriptors, k=count)
        for local_query, candidates in enumerate(rows):
            query = start + local_query
            others = sorted(
                (match for match in candidates if match.trainIdx != query),
                key=lambda match: match.distance,
            )
            if len(others) < 2:
                continue

            keep = len(others)
            for index in range(len(others) - 1):
                next_distance = float(others[index + 1].distance)
                if next_distance == 0:
                    ratio = 0.0 if others[index].distance == 0 else float("inf")
                else:
                    ratio = float(others[index].distance) / next_distance
                if ratio > GENERALIZED_RATIO_THRESHOLD:
                    keep = index + 1
                    break
            matches.extend((query, match.trainIdx) for match in others[:keep])
    matched_pairs = set(matches)
    return [pair for pair in matches if pair[::-1] in matched_pairs]


def _cluster_matches(
    keypoints: list[cv2.KeyPoint], descriptors: np.ndarray
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Cluster matched point locations with Amerini's Ward/IC procedure."""
    matches = _generalized_matches(descriptors)
    if len(matches) < MIN_CLUSTER_MATCHES:
        return {}

    matched_indices = sorted({index for pair in matches for index in pair})
    locations = np.asarray([keypoints[index].pt for index in matched_indices], dtype=np.float64)
    labels = fcluster(
        linkage(locations, method="ward"),
        t=CLUSTER_INCONSISTENCY_THRESHOLD,
        criterion="inconsistent",
        depth=CLUSTER_DEPTH,
    )
    label_by_index = dict(zip(matched_indices, labels))
    clusters: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for query, train in matches:
        source_label = int(label_by_index[query])
        destination_label = int(label_by_index[train])
        # A pair must link two distinct spatial clusters; equal labels are a
        # same-region self-similarity and are not copy-move evidence.
        if source_label == destination_label:
            continue
        clusters[(int(source_label), int(destination_label))].append((query, train))
    return {
        key: pairs
        for key, pairs in clusters.items()
        if len(pairs) >= MIN_CLUSTER_MATCHES
    }


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_distance = float(np.mean(distances))
    if mean_distance <= np.finfo(np.float32).eps:
        return None
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return np.ascontiguousarray((homogeneous @ transform.T)[:, :2]), transform


def _estimate_affine(
    source: np.ndarray, destination: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(source) < MIN_CLUSTER_MATCHES:
        return None
    normalized_source = _normalize_points(source)
    normalized_destination = _normalize_points(destination)
    if normalized_source is None or normalized_destination is None:
        return None
    source_normalized, source_transform = normalized_source
    destination_normalized, destination_transform = normalized_destination
    affine, inlier_mask = cv2.estimateAffine2D(
        source_normalized,
        destination_normalized,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.05,
        maxIters=1000,
        confidence=0.95,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        return None
    normalized_homography = np.vstack((affine, [0.0, 0.0, 1.0]))
    homography = np.linalg.inv(destination_transform) @ normalized_homography @ source_transform
    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    if count < MIN_CLUSTER_MATCHES:
        return None
    return count, source[inliers], destination[inliers], homography


def _is_plausible_affine(homography: np.ndarray) -> bool:
    """Reject numerically unstable matches that imply an impossible transform."""
    try:
        singular_values = np.linalg.svd(homography[:2, :2], compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if not np.all(np.isfinite(singular_values)):
        return False
    smallest, largest = float(singular_values[-1]), float(singular_values[0])
    return (
        smallest >= MIN_AFFINE_SCALE
        and largest <= MAX_AFFINE_SCALE
        and largest / smallest <= MAX_AFFINE_CONDITION
    )


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

        candidate_clusters = _cluster_matches(keypoints, descriptors)
        largest_candidate_matches = max(candidate_clusters.values(), key=len, default=[])
        largest_candidate = len(largest_candidate_matches)
        surviving_matches = sum(len(matches) for matches in candidate_clusters.values())
        candidate_keypoint_indices = {
            index
            for matches in candidate_clusters.values()
            for match in matches
            for index in match
        }
        base_metrics.update({
            "surviving_matches": float(surviving_matches),
            "largest_candidate_region_matches": float(largest_candidate),
            "candidate_region_keypoints": float(len(candidate_keypoint_indices)),
        })

        verified: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]] = []
        verified_cluster_ids: set[int] = set()
        for cluster_ids, matches in sorted(candidate_clusters.items(), key=lambda item: len(item[1]), reverse=True):
            if len(matches) < MIN_CLUSTER_MATCHES:
                continue
            source = np.float32([keypoints[query].pt for query, _ in matches])
            destination = np.float32([keypoints[train].pt for _, train in matches])
            estimated = _estimate_affine(source, destination)
            if estimated is None:
                continue
            inlier_count, source_inliers, destination_inliers, homography = estimated
            if not _is_plausible_affine(homography):
                continue
            verified.append((inlier_count, source_inliers, destination_inliers, homography, cluster_ids))
            verified_cluster_ids.update(cluster_ids)

        overlay = image.copy()
        if not verified:
            score = to_probability(0.0, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
            return DetectorResult(
                self.id, DetectorState.APPLICABLE, score, score >= 0.5, float(config["threshold"]),
                "no_forgery_found: no verified affine cluster was found",
                base_metrics, cv2.resize(overlay, (ctx.width, ctx.height), interpolation=cv2.INTER_LINEAR), _duration(started),
            )

        confident = [item for item in verified if item[0] >= MIN_CONFIDENT_INLIERS]
        hulls: list[np.ndarray] = []
        arrows: list[tuple[np.ndarray, np.ndarray]] = []
        for _, source, destination, _, _ in confident:
            hulls.extend((cv2.convexHull(source), cv2.convexHull(destination)))
            arrows.append((source, destination))
        overlay = _visualization(image, hulls, arrows) if confident else image.copy()
        scale_x = ctx.width / image.shape[1]
        scale_y = ctx.height / image.shape[0]
        best = max(verified, key=lambda item: item[0])
        translation = best[3][:2, 2]
        metrics = {
            **base_metrics,
            "keypoints": float(len(keypoints)),
            "verified_clusters": float(len(verified_cluster_ids)),
            "verified_cluster_pairs": float(len(verified)),
            "inlier_count": float(best[0]),
            "translation_dx": float(translation[0] * scale_x),
            "translation_dy": float(translation[1] * scale_y),
            "affine_scale": float(np.linalg.norm(best[3][0, :2])),
            "affine_rotation_degrees": float(np.degrees(np.arctan2(best[3][1, 0], best[3][0, 0]))),
            "mask_fraction": float(sum(cv2.contourArea(hull) for hull in hulls) / max(1, ctx.width * ctx.height)),
        }
        strongest_inliers = best[0]
        raw = float(len(verified_cluster_ids)) if strongest_inliers >= MIN_CONFIDENT_INLIERS else 0.0
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        if strongest_inliers < MIN_CONFIDENT_INLIERS:
            reason = (
                f"verified local matches were too weak to flag: strongest cluster has "
                f"{strongest_inliers} inliers; {MIN_CONFIDENT_INLIERS} are required"
            )
        else:
            reason = f"verified {len(verified_cluster_ids)} spatial copy-move cluster(s); best cluster has {strongest_inliers} inliers"
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, score >= 0.5, float(config["threshold"]),
            reason,
            metrics, cv2.resize(overlay, (ctx.width, ctx.height), interpolation=cv2.INTER_LINEAR), _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
