"""Public labelled digits, disjoint source splits and controlled caption/duplicate perturbations."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ARMS = {"random-50": ("random", .5), "dedup-50": ("dedup", .5),
        "alignment-50": ("alignment", .5), "combined-50": ("combined", .5),
        "combined-25": ("combined", .25), "combined-75": ("combined", .75)}


def score_threshold(scores, fraction):
    # Adapted from DataComp get_threshold, MIT; see third_party/DataComp.
    return -np.sort(-np.asarray(scores))[int(len(scores)*fraction)]


def fingerprint(image):
    # A gain-invariant binary signature groups controlled near-duplicates.
    bits = np.packbits(image > np.mean(image))
    return bits.tobytes().hex()


def prompt(digit, polarity):
    return f"handwritten digit {digit}, " + ("dark on light" if polarity else "light on dark")


def image_for(images, row):
    arr = images[row["source"]] * row["gain"]
    return 1 - arr if row["polarity"] else arr


def load(seed):
    values = load_digits()
    x = values.images.astype(np.float32)/16
    y = values.target
    ids = np.arange(len(y))
    def split(pool, n):
        return train_test_split(pool, train_size=n, stratify=y[pool], random_state=seed)
    base, rest = split(ids, 480)
    score, rest = split(rest, 250)
    judge, rest = split(rest, 250)
    pool, test = split(rest, 600)
    alignment = KNeighborsClassifier(n_neighbors=9,weights="uniform",algorithm="brute")
    alignment.fit(x[score].reshape(-1,64), y[score])
    scorer = SVC(kernel="rbf",C=5,gamma="scale").fit(x[judge].reshape(-1,64), y[judge])
    rng = np.random.default_rng(seed)
    rows = []
    probs = alignment.predict_proba(x[pool].reshape(-1,64))
    for j, src in enumerate(pool):
        polarity = j%2
        wrong = j%4 == 0 or j%9 == 1
        digit = int((int(y[src])+rng.integers(1,10))%10 if wrong else y[src])
        signature = str(polarity)+"-"+fingerprint(x[src])
        for replica in range(2 if j%3 == 0 else 1):
            rows.append(dict(uid=f"digit-{src}-{polarity}-{replica}",source=int(src),polarity=polarity,
                             digit=digit,true_digit=int(y[src]),gain=1.0 if not replica else .82,
                             signature=signature,align=float(probs[j,digit]),caption=prompt(digit,polarity)))
    return x,y,dict(base=base,alignment=score,judge=judge,pool=pool,test=test),rows,scorer


def select(rows, arm, budget, seed, fraction):
    rule, light_bg = ARMS[arm]
    eligible = list(rows)
    if rule in ("dedup","combined"):
        keep = {}
        for r in eligible:
            keep.setdefault(r["signature"],r)
        eligible = list(keep.values())
    if rule in ("alignment","combined"):
        threshold = score_threshold([r["align"] for r in eligible],fraction)
        # DataComp applies the inclusive comparison before any budget sampling.
        eligible = [r for r in eligible if r["align"] >= threshold]
    result=[]
    rng=np.random.default_rng(seed)
    for polarity,count in [(1,round(budget*light_bg)),(0,budget-round(budget*light_bg))]:
        group=[r for r in eligible if r["polarity"]==polarity]
        if len(group)<count:
            raise ValueError(f"{arm} has {len(group)} eligible samples for a quota of {count}")
        result.extend(group[i] for i in rng.choice(len(group),count,replace=False))
    rng.shuffle(result)
    return result
