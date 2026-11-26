"""Test fixed local-view SigLIP 2 heads and a 50:50 vote with the archived v0.2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from inspector.cabinet_hr import ARCH, DIM, FRACTION, HighResCabinetEncoder, REVISION, SIDE
from scripts.cabinet_final import HEADS, assign_groups, load_images, read_dss, score_set, splits


SEEDS = (42,17,23,71,113)


def head():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=8.0 / DIM, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42,
    ))


def summarize(y, scores):
    metrics = score_set(y, scores)
    dual = (y[:,1:] == 1).all(axis=1)
    metrics["both_found"] = int((scores[dual,1:] >= .50).all(axis=1).sum())
    return metrics


def grouped_delta_intervals(y, groups, control, other):
    """Paired empirical intervals over the held-out visual groups."""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(20261126)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=3000)
    weights = draws[:,inverse]
    def defect_f1(scores):
        predicted = scores[:,1:] >= .5
        truth = y[:,1:].astype(bool)
        tp = weights @ (truth & predicted)
        denominator = weights @ (truth.astype(int) + predicted.astype(int))
        return np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator>0).mean(axis=1)
    delta = defect_f1(other)-defect_f1(control)
    return {"groups":len(unique),"draws":len(draws),"delta_interval_95":np.quantile(delta,[.025,.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--reviewed-groups", type=Path, default=Path("scripts/cabinet_groups.json"))
    parser.add_argument("--control-oof", type=Path, required=True, help="Archived per-seed OOF control predictions with folds")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=("cpu","mps","cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/cabinet-hr-siglip"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write('protocol.json', dict(arch=ARCH, revision=REVISION, side=SIDE, crop_fraction=FRACTION,
        pooled="average of four unit-normalized corner crop embeddings, 72% width and height",
        encoder_precision="float32" if args.device=="cpu" else "float16", device=args.device,
        seeds=SEEDS, folds=5, classifier="train-fold-only balanced StandardScaler/Logistic C=8/d", threshold=.50,
        paired_experiment=["standalone", "equal probability mean with frozen v0.2"], input="pixels only"))
    torch.set_num_threads(args.threads)
    frames = read_dss(args.source)
    images, hashes, meta, _ = load_images(frames)
    assign_groups(frames, hashes, meta, json.loads(args.reviewed_groups.read_text()))
    start = time.monotonic()
    cache = args.out / "private-feature-progress.npz"
    signature = np.array([REVISION, str(FRACTION), str(SIDE), args.device, *meta[:,0]])
    vectors, sample_seconds = [], []
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as prior:
            if np.array_equal(signature, prior["signature"]):
                vectors=list(prior["x"]); sample_seconds=prior["seconds"].tolist()
    load_seconds = None
    if len(vectors) < len(images):
        encoder = HighResCabinetEncoder(args.weights, device=args.device)
        load_seconds = time.monotonic()-start
        print("Encoder loaded", round(load_seconds,2), "cached", len(vectors), flush=True)
    for i,img in enumerate(images[len(vectors):],len(vectors)+1):
        im_start = time.monotonic()
        vectors.append(encoder.embed_image(img))
        sample_seconds.append(time.monotonic()-im_start)
        if i == 1 or i%5==0 or i==len(images):
            temp = cache.with_name(cache.stem + ".tmp.npz")
            np.savez_compressed(temp, signature=signature, x=vectors, seconds=sample_seconds)
            temp.replace(cache)
            print(f"Encoded {i}/{len(images)}, elapsed {time.monotonic()-start:.1f}s",flush=True)
    all_x = np.asarray(vectors)
    np.savez_compressed(args.out/"private-features.npz", x=all_x)
    idx = [i for i,f in enumerate(frames) if f.truth is not None]
    ids = [f"DSS-{i+1:03}" for i in idx]
    y = np.array([frames[i].truth for i in idx]); x=all_x[idx]
    strata=np.array([frames[i].stratum for i in idx]); groups=np.array([frames[i].group for i in idx])
    archived = json.loads(args.control_oof.read_text())
    report={"labeled":len(idx),"total":len(frames),"groups":len(set(groups)),"seeds":{},
            "encoder_runtime":{"device":args.device,"load_s":load_seconds,
                "warm_n":len(sample_seconds)-1,"warm_median_s":float(np.median(sample_seconds[1:])),
                "warm_p95_s":float(np.percentile(sample_seconds[1:],95))}}
    public_primary=[]
    out={}
    for seed in SEEDS:
        fold_for=np.empty(len(y),int); scores=np.zeros(y.shape)
        for fold,(tr,te) in enumerate(splits(strata,groups,5,seed)):
            assert not (set(groups[tr])&set(groups[te]))
            fold_for[te]=fold
            for j in range(3):
                scores[te,j]=head().fit(x[tr],y[tr,j]).predict_proba(x[te])[:,1]
        control=archived[str(seed)]
        assert [r['image_id'] for r in control]==ids
        assert np.array_equal([r['fold'] for r in control],fold_for)
        assert np.array_equal(y, [[r['truth'][h] for h in HEADS] for r in control])
        old=np.array([[r["scores"]["control"][h] for h in HEADS] for r in control])
        out[f"{seed}_hr"]=scores
        out[f"{seed}_blend"]=(scores+old)/2
        if seed==42:
            report["primary_grouped_uncertainty"]={
                name:grouped_delta_intervals(y, groups, old, candidate)
                for name,candidate in (("hr",scores),("v02_hr",(scores+old)/2))
            }
            for i, image_id in enumerate(ids):
                public_primary.append(dict(
                    image_id=image_id,group=int(groups[i]),fold=int(fold_for[i]),
                    truth=dict(zip(HEADS,y[i].tolist(),strict=True)),
                    hr=dict(zip(HEADS,scores[i].tolist(),strict=True)),
                    v02_hr=dict(zip(HEADS,out[f"{seed}_blend"][i].tolist(),strict=True)),
                ))
        report['seeds'][seed]={name:summarize(y,p) for name,p in (
            ("hr",scores),("v02",old),("v02_hr",(scores+old)/2))}
    weights=[];bias=[]
    for j in range(3):
        fitted=head().fit(x,y[:,j]);scaler,logit=(step for _,step in fitted.steps)
        w=logit.coef_[0]/scaler.scale_;weights.append(w);bias.append(logit.intercept_[0]-w@scaler.mean_)
    np.savez_compressed(args.out/'heads.npz',weights=weights,bias=bias)
    np.savez_compressed(args.out/'oof.npz',**out)
    write('oof_primary.json',public_primary)
    from scipy.special import expit
    all_scores=expit(all_x@np.asarray(weights).T+np.asarray(bias))
    write('full_fit_scores.json', [
        dict(image_id=f"DSS-{i+1:03}",scores=dict(zip(HEADS,ps.tolist(),strict=True)))
        for i,ps in enumerate(all_scores)
    ])
    write('benchmark.json',report)
    print({seed:{name:round(m['defect_macro_f1'],4) for name,m in v.items()} for seed,v in report['seeds'].items()})


if __name__=="__main__":
    main()
