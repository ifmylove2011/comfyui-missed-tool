import os
import json
import hashlib
import threading
import numpy as np
import torch
import face_recognition
import onnxruntime as ort
from PIL import Image
from insightface import model_zoo
from insightface.app import FaceAnalysis
from insightface.utils.storage import ensure_available


class FaceRecSimilarityTopK:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_dir": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "./face_cache"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 30}),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "top1_score",
        "top5_avg",
        "top5_min",
        "global_avg",
        "best_match_path",
        "report_text",
        "result_json"
    )

    FUNCTION = "run"
    CATEGORY = "Face Analysis"

    # -------------------------
    # empty result when no face / error
    # -------------------------
    def empty_result(self, reason):
        print(f"[FaceRecSimilarityTopK] {reason}")
        report = f"[WARNING] {reason}"
        result = {"error": reason, "topk": []}
        return (
            0.0, 0.0, 0.0, 0.0, "",
            report,
            json.dumps(result, ensure_ascii=False, indent=2)
        )

    # -------------------------
    # SAFE tensor → numpy
    # -------------------------
    def tensor_to_numpy(self, image):
        img = image[0]

        # 1. force CPU
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

        # 2. float → numpy
        img = img.numpy()

        # 3. scale to uint8
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        # 4. ensure contiguous memory (VERY IMPORTANT for dlib)
        img = np.ascontiguousarray(img)

        return img

    # -------------------------
    # resize short edge 1024
    # -------------------------
    def resize_short_edge_1024(self, img: Image.Image):
        w, h = img.size
        scale = 1024 / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    # -------------------------
    # safe load image from disk
    # -------------------------
    def safe_load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = self.resize_short_edge_1024(img)
        return np.ascontiguousarray(np.array(img))

    # -------------------------
    # embedding cache
    # -------------------------
    def load_or_extract_encoding(self, image_path, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(image_path))[0]
        cache_path = os.path.join(cache_dir, name + ".npy")

        if os.path.exists(cache_path):
            return np.load(cache_path)

        img = self.safe_load_image(image_path)

        try:
            encs = face_recognition.face_encodings(img)
        except Exception as e:
            print(f"[ERROR] face crash on {image_path}: {e}")
            return None

        if len(encs) == 0:
            return None

        emb = encs[0]
        np.save(cache_path, emb)
        return emb

    # -------------------------
    # cosine similarity
    # -------------------------
    def cosine(self, a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # -------------------------
    # main
    # -------------------------
    def run(self, image, source_dir, cache_dir, top_k):

        if not os.path.exists(source_dir):
            return self.empty_result("source_dir not found")

        # -------------------------
        # generate embedding
        # -------------------------
        gen_np = self.tensor_to_numpy(image)

        try:
            gen_encs = face_recognition.face_encodings(gen_np)
        except Exception as e:
            return self.empty_result(f"face_recognition crashed on generated image: {e}")

        if len(gen_encs) == 0:
            return self.empty_result("No face detected in generated image")

        gen_emb = gen_encs[0]

        # -------------------------
        # load source images
        # -------------------------
        files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        if len(files) == 0:
            return self.empty_result("No source images found")

        scores = []
        total_sum = 0.0
        valid_count = 0

        # -------------------------
        # similarity compute
        # -------------------------
        for f in files:
            path = os.path.join(source_dir, f)
            emb = self.load_or_extract_encoding(path, cache_dir)

            if emb is None:
                continue

            sim = self.cosine(gen_emb, emb)

            scores.append((f, sim))
            total_sum += sim
            valid_count += 1

        if valid_count == 0:
            return self.empty_result("No valid face encodings in source images")

        # -------------------------
        # sort
        # -------------------------
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = scores[:top_k]

        # -------------------------
        # metrics
        # -------------------------
        top1 = topk[0][1]
        top5_avg = sum(x[1] for x in topk) / len(topk)
        top5_min = min(x[1] for x in topk)
        global_avg = total_sum / valid_count
        best_match = topk[0][0]

        # -------------------------
        # report
        # -------------------------
        top5_lines = []
        topk_struct = []

        for i, (fname, score) in enumerate(topk):
            top5_lines.append(f"{i+1}. {fname} -> {score:.4f}")

            topk_struct.append({
                "rank": i + 1,
                "path": fname,
                "score": float(score)
            })

        result_json = {
            "top1_score": float(top1),
            "top5_avg": float(top5_avg),
            "top5_min": float(top5_min),
            "global_avg": float(global_avg),
            "best_match": {
                "path": best_match,
                "score": float(top1)
            },
            "topk": topk_struct,
            "settings": {
                "top_k": int(top_k),
                "embedding": "face_recognition_dlib_128d",
                "similarity": "cosine"
            }
        }

        json_str = json.dumps(result_json, ensure_ascii=False, indent=2)

        report_text = (
            f"Top1: {top1:.4f}\n"
            f"Top5 Avg: {top5_avg:.4f}\n"
            f"Top5 Min: {top5_min:.4f}\n"
            f"Global Avg: {global_avg:.4f}\n\n"
            f"Best Match: {best_match}\n\n"
            + "\n".join(top5_lines)
        )

        return (
            float(top1),
            float(top5_avg),
            float(top5_min),
            float(global_avg),
            best_match,
            report_text,
            json_str
        )



class InsightFaceSimilarityTopK:

    _app = None

    @classmethod
    def get_app(cls):
        if cls._app is None:
            cls._app = FaceAnalysis(name="buffalo_l")
            cls._app.prepare(ctx_id=0, det_size=(640, 640))
        return cls._app

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_dir": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "./insight_cache"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "top1_score",
        "top5_avg",
        "top5_min",
        "global_avg",
        "best_match_path",
        "report_text",
        "result_json"
    )

    FUNCTION = "run"
    CATEGORY = "Face Analysis"

    # -------------------------
    # utils
    # -------------------------
    def tensor_to_np(self, image):
        img = image[0]
        img = (img * 255).cpu().numpy().astype(np.uint8)
        return img

    def cosine(self, a, b):
        return float(np.dot(a, b))

    def load_cache(self, path):
        return np.load(path)

    def save_cache(self, path, emb):
        np.save(path, emb)

    # -------------------------
    # extract embedding (InsightFace)
    # -------------------------
    def get_embedding(self, app, img):
        faces = app.get(img)
        if len(faces) == 0:
            return None

        # 取最大脸（最稳定）
        faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
        return faces[0].normed_embedding

    # -------------------------
    # main
    # -------------------------
    def run(self, image, source_dir, cache_dir, top_k):

        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(source_dir):
            raise Exception("source_dir not found")

        app = self.get_app()

        # -------------------------
        # generated image embedding
        # -------------------------
        gen_np = self.tensor_to_np(image)
        gen_emb = self.get_embedding(app, gen_np)

        if gen_emb is None:
            raise Exception("No face detected in generated image")

        # -------------------------
        # source images
        # -------------------------
        files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        scores = []
        total = 0.0
        valid = 0

        # -------------------------
        # compare
        # -------------------------
        for f in files:
            path = os.path.join(source_dir, f)
            cache_path = os.path.join(cache_dir, f + ".npy")

            if os.path.exists(cache_path):
                emb = self.load_cache(cache_path)
            else:
                img = Image.open(path).convert("RGB")
                img = np.array(img)
                emb = self.get_embedding(app, img)

                if emb is not None:
                    self.save_cache(cache_path, emb)

            if emb is None:
                continue

            sim = self.cosine(gen_emb, emb)

            scores.append((f, sim))
            total += sim
            valid += 1

        if valid == 0:
            raise Exception("No valid faces in source images")

        # -------------------------
        # sort
        # -------------------------
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = scores[:top_k]

        # -------------------------
        # metrics
        # -------------------------
        top1 = topk[0][1]
        top5_avg = sum(x[1] for x in topk) / len(topk)
        top5_min = min(x[1] for x in topk)
        global_avg = total / valid

        best = topk[0][0]

        # -------------------------
        # topk struct
        # -------------------------
        topk_struct = []
        topk_paths = []
        lines = []

        for i, (f, s) in enumerate(topk):
            topk_struct.append({
                "rank": i + 1,
                "path": f,
                "score": float(s)
            })

            topk_paths.append(f)
            lines.append(f"{i + 1}. {f} -> {s:.4f}")

        # -------------------------
        # JSON
        # -------------------------
        result = {
            "top1_score": float(top1),
            "top5_avg": float(top5_avg),
            "top5_min": float(top5_min),
            "global_avg": float(global_avg),

            "best_match": {
                "path": best,
                "score": float(top1)
            },

            "topk": topk_struct,
            "topk_paths": topk_paths,

            "backend": "insightface_arcface",
            "model": "buffalo_l"
        }

        json_str = json.dumps(result, ensure_ascii=False, indent=2)

        report = (
                f"Top1: {top1:.4f}\n"
                f"Top5 Avg: {top5_avg:.4f}\n"
                f"Top5 Min: {top5_min:.4f}\n"
                f"Global Avg: {global_avg:.4f}\n\n"
                f"Best Match: {best}\n\n"
                f"TopK:\n"
                + "\n".join(lines)
        )

        return (
            float(top1),
            float(top5_avg),
            float(top5_min),
            float(global_avg),
            best,
            report,
            json_str
        )


class HybridFaceSimilarityTopK:
    """
    Face-recognition ranking with ArcFace as a penalty for obvious mismatches.
    The main score is face_recognition cosine similarity. ArcFace only reduces
    that score when its similarity is below arcface_penalty_threshold.
    """

    _app = None

    @classmethod
    def get_app(cls):
        if cls._app is None:
            cls._app = FaceAnalysis(name="buffalo_l")
            cls._app.prepare(ctx_id=0, det_size=(640, 640))
        return cls._app

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_dir": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "./hybrid_face_cache"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 30}),
                "arcface_penalty_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "penalty_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "top1_hybrid_score",
        "top1_arc_score",
        "top5_hybrid_avg",
        "top5_arc_avg",
        "global_hybrid_avg",
        "best_match_path",
        "report_text",
        "result_json"
    )

    FUNCTION = "run"
    CATEGORY = "Face Analysis"

    # -------------------------
    # tensor → numpy (RGB uint8)
    # -------------------------
    def tensor_to_numpy(self, image):
        img = image[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        img = img.numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    # -------------------------
    # load + resize short edge to 1024 → RGB numpy
    # -------------------------
    def safe_load_image_rgb(self, path):
        pil = Image.open(path).convert("RGB")
        w, h = pil.size
        scale = 1024 / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        return np.ascontiguousarray(np.array(pil))

    # -------------------------
    # pick largest face by bbox area
    # -------------------------
    def largest_face(self, faces):
        return max(
            faces,
            key=lambda f: max(0, int(f.bbox[2] - f.bbox[0]))
            * max(0, int(f.bbox[3] - f.bbox[1])),
            default=None,
        )

    # -------------------------
    # extract both face_recognition + ArcFace embeddings from RGB image
    # ArcFace detects first, its bbox is reused for face_recognition
    # -------------------------
    def extract_embeddings(self, app, rgb_img):
        bgr = np.ascontiguousarray(rgb_img[:, :, ::-1])
        face = self.largest_face(app.get(bgr))
        if face is None:
            return None, None

        h, w = rgb_img.shape[:2]
        left = max(0, min(w, int(face.bbox[0])))
        top = max(0, min(h, int(face.bbox[1])))
        right = max(left + 1, min(w, int(face.bbox[2])))
        bottom = max(top + 1, min(h, int(face.bbox[3])))

        encodings = face_recognition.face_encodings(
            rgb_img,
            known_face_locations=[(top, right, bottom, left)],
            num_jitters=1,
        )
        if not encodings:
            return None, None

        arc_emb = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(arc_emb)
        if norm == 0:
            return None, None
        arc_emb /= norm

        return encodings[0], arc_emb

    # -------------------------
    # load or extract + cache both embeddings (.npz)
    # -------------------------
    def load_or_extract_hybrid(self, app, image_path, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(image_path))[0]
        cache_path = os.path.join(cache_dir, name + ".npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["face_encoding"], data["arc_embedding"]

        rgb = self.safe_load_image_rgb(image_path)
        face_enc, arc_emb = self.extract_embeddings(app, rgb)
        if face_enc is None:
            return None, None

        np.savez(cache_path, face_encoding=face_enc, arc_embedding=arc_emb)
        return face_enc, arc_emb

    # -------------------------
    # pair scores: face_recognition cosine, ArcFace cosine, hybrid
    # -------------------------
    def pair_scores(self, gen_face, gen_arc, src_face, src_arc, threshold, factor):
        face_score = float(np.dot(gen_face, src_face) / (
            np.linalg.norm(gen_face) * np.linalg.norm(src_face)
        ))
        arc_score = float(np.dot(gen_arc, src_arc))
        penalty = max(0.0, (1.0 - arc_score) * factor) if arc_score < threshold else 0.0
        return face_score, arc_score, face_score - penalty

    # -------------------------
    # main
    # -------------------------
    def run(self, image, source_dir, cache_dir, top_k,
            arcface_penalty_threshold, penalty_factor):

        if not os.path.exists(source_dir):
            raise Exception("source_dir not found")

        app = self.get_app()

        # -------------------------
        # generated image embeddings
        # -------------------------
        gen_rgb = self.tensor_to_numpy(image)
        gen_face_enc, gen_arc_emb = self.extract_embeddings(app, gen_rgb)

        if gen_face_enc is None:
            raise Exception("No face detected in generated image")

        # -------------------------
        # source images
        # -------------------------
        files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        if len(files) == 0:
            raise Exception("No source images found")

        results = []

        # -------------------------
        # compute scores
        # -------------------------
        for f in files:
            path = os.path.join(source_dir, f)
            face_enc, arc_emb = self.load_or_extract_hybrid(app, path, cache_dir)

            if face_enc is None:
                continue

            face_score, arc_score, hybrid_score = self.pair_scores(
                gen_face_enc, gen_arc_emb, face_enc, arc_emb,
                arcface_penalty_threshold, penalty_factor
            )

            results.append({
                "filename": f,
                "face_score": face_score,
                "arc_score": arc_score,
                "hybrid_score": hybrid_score,
            })

        if len(results) == 0:
            raise Exception("No valid face encodings in source images")

        # -------------------------
        # sort by hybrid score
        # -------------------------
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        topk = results[:top_k]

        # -------------------------
        # metrics
        # -------------------------
        top1_hybrid = topk[0]["hybrid_score"]
        top1_arc = topk[0]["arc_score"]
        top5_hybrid_avg = sum(x["hybrid_score"] for x in topk) / len(topk)
        top5_arc_avg = sum(x["arc_score"] for x in topk) / len(topk)
        global_hybrid_avg = sum(x["hybrid_score"] for x in results) / len(results)
        best_match = topk[0]["filename"]

        # -------------------------
        # report
        # -------------------------
        lines = []
        topk_struct = []

        for i, r in enumerate(topk):
            lines.append(
                f"{i+1}. {r['filename']} -> hybrid={r['hybrid_score']:.4f} "
                f"(face={r['face_score']:.4f}, arc={r['arc_score']:.4f})"
            )

            topk_struct.append({
                "rank": i + 1,
                "path": r["filename"],
                "hybrid_score": float(r["hybrid_score"]),
                "face_score": float(r["face_score"]),
                "arc_score": float(r["arc_score"]),
            })

        result_json = {
            "top1_hybrid_score": float(top1_hybrid),
            "top1_arc_score": float(top1_arc),
            "top5_hybrid_avg": float(top5_hybrid_avg),
            "top5_arc_avg": float(top5_arc_avg),
            "global_hybrid_avg": float(global_hybrid_avg),
            "best_match": {
                "path": best_match,
                "hybrid_score": float(top1_hybrid),
                "arc_score": float(top1_arc),
            },
            "topk": topk_struct,
            "settings": {
                "top_k": int(top_k),
                "arcface_penalty_threshold": float(arcface_penalty_threshold),
                "penalty_factor": float(penalty_factor),
                "face_embedding": "face_recognition_dlib_128d",
                "arc_embedding": "insightface_buffalo_l_512d",
                "similarity": "cosine",
            },
        }

        json_str = json.dumps(result_json, ensure_ascii=False, indent=2)

        report_text = (
            f"Top1 Hybrid: {top1_hybrid:.4f}\n"
            f"Top1 Arc: {top1_arc:.4f}\n"
            f"Top5 Hybrid Avg: {top5_hybrid_avg:.4f}\n"
            f"Top5 Arc Avg: {top5_arc_avg:.4f}\n"
            f"Global Hybrid Avg: {global_hybrid_avg:.4f}\n\n"
            f"Best Match: {best_match}\n\n"
            + "\n".join(lines)
        )

        return (
            float(top1_hybrid),
            float(top1_arc),
            float(top5_hybrid_avg),
            float(top5_arc_avg),
            float(global_hybrid_avg),
            best_match,
            report_text,
            json_str
        )


class FaceIdentitySimilarity:
    """Compare ComfyUI images with an identity centroid made from reference photos.

    The InsightFace application is process-wide and initialized lazily, so loading
    this node repeatedly does not reload the antelopev2 ONNX models.
    """

    _app = None
    _app_lock = threading.Lock()
    _inference_lock = threading.Lock()
    MODEL_NAME = "antelopev2"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "reference_dir": ("STRING", {"default": ""}),
            "cache_dir": ("STRING", {"default": "./face_identity_cache"}),
            "top_k": ("INT", {"default": 5, "min": 1, "max": 30}),
            "pass_threshold": ("FLOAT", {"default": 0.55, "min": -1.0, "max": 1.0, "step": 0.01}),
            "outlier_threshold": ("FLOAT", {"default": 0.30, "min": -1.0, "max": 1.0, "step": 0.01}),
            "loo_threshold": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
            "closeup_area_ratio": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 1.0, "step": 0.005}),
            "closeup_expand": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            "clean_references": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("similarity", "passed", "best_match", "json", "face_crop", "report_text")
    FUNCTION = "run"
    CATEGORY = "Face Analysis"

    @classmethod
    def get_app(cls):
        if cls._app is None:
            with cls._app_lock:
                if cls._app is None:
                    available = ort.get_available_providers()
                    if "CUDAExecutionProvider" not in available:
                        raise RuntimeError(
                            "FaceIdentitySimilarity requires CUDAExecutionProvider, "
                            f"but ONNX Runtime only reports: {available}. Install a "
                            "CUDA-compatible onnxruntime-gpu build and restart ComfyUI."
                        )
                    # Deliberately do not fall back to CPU: this node requires CUDA.
                    # InsightFace 1.0.1 creates an ONNX session for every file before
                    # applying allowed_modules. Build a minimal FaceAnalysis object
                    # directly so auxiliary antelopev2 sessions can neither alter
                    # provider selection nor consume resources during initialization.
                    model_dir = ensure_available("models", cls.MODEL_NAME, root="~/.insightface")
                    model_files = {
                        "recognition": os.path.join(model_dir, "glintr100.onnx"),
                        "detection": os.path.join(model_dir, "scrfd_10g_bnkps.onnx"),
                    }
                    for model_path in model_files.values():
                        if not os.path.isfile(model_path):
                            raise RuntimeError("Missing antelopev2 model: " + model_path)

                    # Load recognition first. This is the session that produces the
                    # required 512D identity embedding.
                    models = {
                        task: model_zoo.get_model(
                            model_files[task], providers=["CUDAExecutionProvider"]
                        )
                        for task in ("recognition", "detection")
                    }
                    if any(model is None for model in models.values()):
                        raise RuntimeError("Could not construct antelopev2 detection/recognition models")
                    app = FaceAnalysis.__new__(FaceAnalysis)
                    app.model_dir = model_dir
                    app.models = models
                    app.det_model = models["detection"]
                    app.prepare(ctx_id=0, det_size=(640, 640))
                    # Some antelopev2 auxiliary heads (notably landmark_3d_68)
                    # legitimately run on CPU when their graph contains operators
                    # unsupported by the installed CUDA EP. They are not used to
                    # produce the identity embedding. Enforce CUDA only for the two
                    # models on the identity path: face detection and recognition.
                    cuda_required_models = {"detection", "recognition"}
                    for model_name, model in app.models.items():
                        if model_name not in cuda_required_models:
                            continue
                        session = getattr(model, "session", None)
                        providers = session.get_providers() if session is not None else []
                        if not providers or providers[0] != "CUDAExecutionProvider":
                            raise RuntimeError(
                                f"antelopev2 model '{model_name}' did not initialize on CUDA "
                                f"(active providers: {providers})."
                            )
                    cls._app = app
        return cls._app

    @classmethod
    def detect_faces(cls, app, rgb):
        # FaceAnalysis owns shared ONNX sessions; serialize access because ComfyUI may
        # execute separate workflows concurrently against this process-wide instance.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        with cls._inference_lock:
            return app.get(bgr)

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

    @staticmethod
    def normalize(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else None

    @staticmethod
    def largest_face(faces):
        return max(faces, key=lambda f: max(0.0, float(f.bbox[2] - f.bbox[0])) *
                   max(0.0, float(f.bbox[3] - f.bbox[1])), default=None)

    @staticmethod
    def tensor_image_to_rgb(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def load_rgb(path):
        with Image.open(path) as pil:
            return np.ascontiguousarray(np.asarray(pil.convert("RGB")))

    @staticmethod
    def expanded_box(bbox, width, height, scale):
        x1, y1, x2, y2 = map(float, bbox)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_w, half_h = max(1.0, (x2 - x1) * scale / 2.0), max(1.0, (y2 - y1) * scale / 2.0)
        return (max(0, int(cx - half_w)), max(0, int(cy - half_h)),
                min(width, int(cx + half_w)), min(height, int(cy + half_h)))

    def embedding_and_crop(self, app, rgb, area_ratio, expand):
        h, w = rgb.shape[:2]
        faces = self.detect_faces(app, rgb)
        face = self.largest_face(faces)
        if face is None:
            return None, None, {"detected": False}

        fw = max(0.0, float(face.bbox[2] - face.bbox[0]))
        fh = max(0.0, float(face.bbox[3] - face.bbox[1]))
        small_face = (fw * fh) / max(1.0, float(w * h)) < area_ratio
        crop_box = self.expanded_box(face.bbox, w, h, expand)
        crop = rgb[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        embedding = self.normalize(getattr(face, "normed_embedding", None))
        rescued = False
        if small_face and crop.size:
            rescue_faces = self.detect_faces(app, crop)
            rescue_face = self.largest_face(rescue_faces)
            if rescue_face is not None:
                embedding = self.normalize(getattr(rescue_face, "normed_embedding", None))
                rescued = embedding is not None
        if embedding is None or embedding.shape != (512,):
            return None, crop, {"detected": True, "closeup_rescue": rescued}
        return embedding.astype(np.float32), crop, {
            "detected": True, "closeup_rescue": rescued,
            "face_area_ratio": float((fw * fh) / max(1.0, w * h)),
        }

    def cached_reference(self, app, path, cache_dir, area_ratio, expand):
        stat = os.stat(path)
        cache_identity = (
            os.path.abspath(path), stat.st_mtime_ns, stat.st_size, self.MODEL_NAME,
            float(area_ratio), float(expand), 512,
        )
        key = hashlib.sha1(repr(cache_identity).encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, key + ".npy")
        if os.path.isfile(cache_path):
            emb = self.normalize(np.load(cache_path))
            if emb is not None and emb.shape == (512,):
                return emb
        emb, _, _ = self.embedding_and_crop(app, self.load_rgb(path), area_ratio, expand)
        if emb is not None:
            np.save(cache_path, emb)
        return emb

    def clean_embeddings(self, records, outlier_threshold, loo_threshold):
        if len(records) < 2:
            return records, []
        all_emb = np.stack([r["embedding"] for r in records])
        centroid = self.normalize(np.mean(all_emb, axis=0))
        initial = [self.cosine(e, centroid) for e in all_emb]
        kept = [r for r, score in zip(records, initial) if score >= outlier_threshold]
        removed = [r["name"] for r, score in zip(records, initial) if score < outlier_threshold]
        if len(kept) >= 2:
            loo_kept = []
            for i, record in enumerate(kept):
                others = [r["embedding"] for j, r in enumerate(kept) if i != j]
                loo_centroid = self.normalize(np.mean(others, axis=0))
                if loo_centroid is not None and self.cosine(record["embedding"], loo_centroid) >= loo_threshold:
                    loo_kept.append(record)
                else:
                    removed.append(record["name"])
            kept = loo_kept
        return kept, removed

    def run(self, image, reference_dir, cache_dir, top_k, pass_threshold,
            outlier_threshold, loo_threshold, closeup_area_ratio,
            closeup_expand, clean_references):
        if not os.path.isdir(reference_dir):
            raise RuntimeError("reference_dir not found: " + str(reference_dir))
        os.makedirs(cache_dir, exist_ok=True)
        app = self.get_app()
        extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        paths = [os.path.join(reference_dir, f) for f in sorted(os.listdir(reference_dir))
                 if f.lower().endswith(extensions) and os.path.isfile(os.path.join(reference_dir, f))]
        records, skipped = [], []
        for path in paths:
            try:
                emb = self.cached_reference(app, path, cache_dir, closeup_area_ratio, closeup_expand)
                if emb is None:
                    skipped.append(os.path.basename(path))
                else:
                    records.append({"name": os.path.basename(path), "path": path, "embedding": emb})
            except Exception as exc:
                skipped.append({"name": os.path.basename(path), "error": str(exc)})
        if not records:
            raise RuntimeError("No valid face embeddings in reference_dir")
        removed = []
        if clean_references:
            records, removed = self.clean_embeddings(records, outlier_threshold, loo_threshold)
        if not records:
            raise RuntimeError(
                "Reference cleaning removed every embedding; lower outlier_threshold/"
                "loo_threshold or disable clean_references."
            )
        centroid = self.normalize(np.mean([r["embedding"] for r in records], axis=0))
        if centroid is None:
            raise RuntimeError("Could not calculate identity centroid")

        crops, per_image = [], []
        for item in image:
            rgb = self.tensor_image_to_rgb(item)
            emb, crop, meta = self.embedding_and_crop(app, rgb, closeup_area_ratio, closeup_expand)
            if crop is None or crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            if emb is None:
                per_image.append({"similarity": 0.0, "passed": False, "best_match": "", "error": "No face detected", **meta})
            else:
                scores = sorted(
                    ((r["name"], self.cosine(emb, r["embedding"])) for r in records),
                    key=lambda x: x[1], reverse=True,
                )
                selected_topk = scores[:min(int(top_k), len(scores))]
                best_name, best_score = selected_topk[0]
                similarity = self.cosine(emb, centroid)
                per_image.append({"similarity": similarity, "passed": similarity >= pass_threshold,
                                  "best_match": best_name, "best_match_similarity": best_score,
                                  "topk_avg_similarity": float(np.mean([score for _, score in selected_topk])),
                                  "topk": [{"rank": rank, "name": name, "similarity": score}
                                           for rank, (name, score) in enumerate(selected_topk, 1)],
                                  **meta})
            pil = Image.fromarray(crop).convert("RGB")
            pil.thumbnail((256, 256), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (256, 256))
            canvas.paste(pil, ((256 - pil.width) // 2, (256 - pil.height) // 2))
            crops.append(np.asarray(canvas, dtype=np.float32) / 255.0)

        result = {"similarity": float(np.mean([x["similarity"] for x in per_image])),
                  "passed": bool(all(x["passed"] for x in per_image)),
                  "best_match": max(per_image, key=lambda x: x["similarity"])["best_match"],
                  "per_image": per_image, "reference_count": len(records),
                  "removed_by_cleaning": removed, "skipped": skipped,
                  "model": self.MODEL_NAME, "embedding_dimensions": 512,
                  "execution_provider": "CUDAExecutionProvider", "similarity_metric": "cosine",
                  "settings": {"top_k": int(top_k), "pass_threshold": pass_threshold,
                                "outlier_threshold": outlier_threshold,
                                "loo_threshold": loo_threshold, "clean_references": clean_references}}

        valid_results = [item for item in per_image if "error" not in item]
        if valid_results:
            best_result = max(valid_results, key=lambda item: item["best_match_similarity"])
            all_topk_scores = [entry["similarity"] for item in valid_results for entry in item["topk"]]
            report_lines = [
                f"Final Similarity: {result['similarity']:.4f}",
                f"Top1 Similarity: {best_result['best_match_similarity']:.4f}",
                f"Top{int(top_k)} Avg Similarity: {float(np.mean(all_topk_scores)):.4f}",
                "",
                f"Best Match: {best_result['best_match']}",
                "",
            ]
            if len(valid_results) == 1:
                report_lines.extend(
                    f"{entry['rank']}. {entry['name']} -> {entry['similarity']:.4f}"
                    for entry in valid_results[0]["topk"]
                )
            else:
                for image_index, item in enumerate(valid_results, 1):
                    report_lines.append(f"Image {image_index}:")
                    report_lines.extend(
                        f"{entry['rank']}. {entry['name']} -> {entry['similarity']:.4f}"
                        for entry in item["topk"]
                    )
                    report_lines.append("")
            report_text = "\n".join(report_lines).rstrip()
        else:
            report_text = (
                f"Final Similarity: {result['similarity']:.4f}\n"
                "Top1 Similarity: 0.0000\n"
                f"Top{int(top_k)} Avg Similarity: 0.0000\n\n"
                "Best Match: \n\nNo valid face detected."
            )
        return (result["similarity"], result["passed"], result["best_match"],
                json.dumps(result, ensure_ascii=False, indent=2),
                torch.from_numpy(np.stack(crops)), report_text)
