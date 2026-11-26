"""Paired, grouped multiple-instance ablation for the released v0.4 defect heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from inspector.cabinet_hr import DIM, HighResCabinetEncoder, REVISION
from scripts.cabinet_final import HEADS, assign_groups, load_images, read_dss, score_set, splits

SEEDS = (42, 17, 23, 71, 113)
TEMPERATURE = 3.0
C = 8.0 / DIM


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def extract_bags(images, hashes, cache, device):
    signature = np.array([REVISION, device, *hashes])
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as prior:
            if np.array_equal(prior["signature"], signature):
                return prior["bags"]
    encoder = HighResCabinetEncoder(device=device)
    patches, data = [], []
    def capture(model, args, output):
        patches.extend(torch.nn.functional.normalize(output.detach().float(), dim=1).cpu().numpy())
    handle = encoder.model.register_forward_hook(capture)
    try:
        for i, image in enumerate(images):
            patches.clear()
            expected_mean = encoder.embed_image(image)
            bag = np.array(patches)
            if bag.shape != (4, DIM) or not np.allclose(expected_mean, bag.mean(axis=0), atol=1.e-7):
                raise RuntimeError("Unexpected HR crop embeddings.")
            data.append(bag)
            if (i + 1) % 10 == 0:
                print(f"Encoded {i + 1}/{len(images)} independent bags", flush=True)
    finally:
        handle.remove()
    bags = np.array(data)
    np.savez_compressed(cache, signature=signature, bags=bags)
    return bags


def pool(logits):
    return (logsumexp(logits * TEMPERATURE, axis=1) - np.log(logits.shape[1])) / TEMPERATURE


def fit_mil(bags, labels):
    scaler = StandardScaler().fit(bags.reshape(-1, DIM))
    x = ((bags - scaler.mean_) / scaler.scale_).astype(np.float64)
    init = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=300,
                              random_state=42).fit(x.mean(axis=1), labels)
    count = np.bincount(labels, minlength=2)
    observation_weight = len(labels) / (2 * count[labels])
    def objective(params):
        patch_logits = x @ params[:-1]
        pooled = pool(patch_logits) + params[-1]
        residual = observation_weight * (expit(pooled) - labels)
        attention = softmax(TEMPERATURE * patch_logits, axis=1)
        feature_grad = np.einsum("nbd,nb,n->d", x, attention, residual, optimize=True)
        # As in the matched liblinear control, regularize the intercept as a synthetic feature.
        loss = observation_weight @ (np.logaddexp(0, pooled) - labels * pooled) + params @ params / (2*C)
        grad = np.append(feature_grad, residual.sum()) + params / C
        return loss, grad
    out = minimize(objective, np.r_[init.coef_[0], init.intercept_[0]], jac=True,
                   method="L-BFGS-B", options={"maxiter": 200, "ftol": 1.e-10, "gtol": 1.e-6})
    if not out.success:
        raise RuntimeError(out.message)
    weights = out.x[:-1] / scaler.scale_
    bias = out.x[-1] - scaler.mean_ @ weights
    return weights, bias


def metrics(y, predictions):
    m = score_set(y, predictions)
    dual_truth = y[:, 1:].all(axis=1)
    dual_guess = (predictions[:, 1:] >= .50).all(axis=1)
    m["true_dual"] = int((dual_truth & dual_guess).sum())
    m["false_dual"] = int((~dual_truth & dual_guess).sum())
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--control", type=Path, default=Path("runs/final-cabinet-v0.4-hr-roi/oof_by_seed.json"))
    p.add_argument("--original", type=Path, default=Path("runs/final-cabinet-experiment/oof_comparison.json"))
    p.add_argument("--out", type=Path, default=Path("runs/cabinet-mil"))
    p.add_argument("--device", default="mps", choices=("mps","cpu","cuda"))
    a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    write(a.out/"public-protocol.json", dict(revision=REVISION, seeds=SEEDS, methods=["v0.4","v0.2 and MIL 50:50"],
          windows="four 72% corner windows in released HR branch", pooling="log mean exp", temperature=TEMPERATURE,
          C=C, normalization="outer training patches only", labels="bag-level independent binary",
          decision_threshold=.5, promote_if="better macro-F1 on all five seeds and better primary true dual count"))
    torch.set_num_threads(2)
    frames=read_dss(a.source)
    images, hashes, meta, _=load_images(frames)
    assign_groups(frames, hashes, meta, json.loads(Path("scripts/cabinet_groups.json").read_text()))
    data=extract_bags(images,meta[:,0],a.out/"private-bags.npz",a.device)
    idx=[i for i,f in enumerate(frames) if f.truth is not None]
    x=data[idx]; y=np.asarray([frames[i].truth for i in idx])
    ids=[f"DSS-{i+1:03}" for i in idx];g=np.array([frames[i].group for i in idx])
    strata=np.array([frames[i].stratum for i in idx])
    version=json.loads(a.control.read_text());orig=json.loads(a.original.read_text())
    all_scores={}; results={}
    for seed in SEEDS:
        current=version[str(seed)]; original=orig[str(seed)]
        for rows in (current,original):
            assert [r["image_id"] for r in rows] == ids and np.array_equal([r["group"] for r in rows], g)
            assert np.array_equal(y,[[r["truth"][key] for key in HEADS] for r in rows])
        base=np.array([[r["object_type_score"] if r["object_type"]=="telecom_cabinet" else 1-r["object_type_score"],
                        r["scores"]["mechanical"],r["scores"]["coating"]] for r in current])
        original_scores=np.array([[r["scores"]["control"][h] for h in HEADS] for r in original])
        mil=base.copy()
        for fold,(tr,te) in enumerate(splits(strata,g,5,seed)):
            assert all(current[i]["fold"]==original[i]["fold"]==fold for i in te)
            assert set(g[tr]).isdisjoint(g[te])
            for j in (1,2):
                w,b=fit_mil(x[tr],y[tr,j])
                mil[te,j]=expit(pool(x[te] @ w)+b)
        candidate=base.copy();candidate[:,1:] = (mil[:,1:]+original_scores[:,1:])/2
        results[seed]={"v04":metrics(y,base),"mil_blend":metrics(y,candidate),
                       "mil_standalone_defects":metrics(y,mil)}
        all_scores[seed]=[dict(image_id=id, group=int(g[i]), fold=current[i]["fold"],
                       truth=dict(zip(HEADS,y[i].tolist())), scores=dict(zip(HEADS,candidate[i].tolist())),
                       control=dict(zip(HEADS,base[i].tolist()))) for i,id in enumerate(ids)]
        print(seed,[(k,round(v["defect_macro_f1"],4),v["true_dual"],v["false_dual"]) for k,v in results[seed].items()],flush=True)
    write(a.out/"public-oof.json",all_scores)
    improved = all(m["mil_blend"]["defect_macro_f1"] > m["v04"]["defect_macro_f1"] for m in results.values())
    added_dual = results[42]["mil_blend"]["true_dual"] > results[42]["v04"]["true_dual"]
    means = {name: float(np.mean([metrics[name]["defect_macro_f1"] for metrics in results.values()]))
             for name in ("v04", "mil_blend", "mil_standalone_defects")}
    write(a.out/"public-benchmark.json",dict(labeled=len(idx),groups=len(set(g)),seeds=results,
          mean_defect_macro_f1=means,decision="promote_mil" if improved and added_dual else "retain_v04"))


if __name__=="__main__":
    main()
