"""Zero-shot relevance filter for harvested photos (open_clip ViT-B/32, LAION-400M weights; filtering only)."""
import torch, open_clip, numpy as np, cv2
from PIL import Image
torch.set_num_threads(8)
MODEL, _, PRE = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="/home/claude/models/vitb32_laion400m.pt")
TOK = open_clip.get_tokenizer("ViT-B-32-quickgelu")
MODEL.eval()
POS = {
 "parking": ["a photo of a snow covered parking lot", "a parking lot with snow and parked cars", "a snowy parking lot seen from above",
             "a plowed parking lot with snow piles"],
 "street": ["a photo of a snowy city street", "a road covered with snow and tire tracks", "a street with slush and dirty snow",
            "a residential street after snowfall", "snowbanks along a plowed road"],
 "sidewalk": ["a snowy sidewalk in a city", "a snow covered courtyard between buildings"],
 "roof": ["snow on a building roof", "a flat roof covered with snow"],
 "ground_closeup": ["close-up of snow on asphalt", "tire tracks in snow, close-up", "dirty slushy snow on the ground"],
 "clearing": ["a snow plow clearing a road", "a tractor removing snow in a parking lot"],
}
NEG = ["an ice hockey player", "a portrait of a person", "a group of people", "a snowy mountain landscape",
       "a snowy forest", "a lake or river", "a satellite image", "a map or diagram", "an indoor scene", "a car in a garage",
       "a building facade without snow", "a sports game", "an animal", "a sunset sky", "a toy or model", "fire at night",
       "a summer landscape", "a black and white historical photo", "a snowy mountain peak", "a forest road in winter",
       "orange sand with tire tracks", "a beach", "ski slope with skiers", "a dog sled", "a snowman", "a bridge over a river",
       "a church or monument in winter", "trees covered with snow"]
# roof set: what the "photo from a taller building" module needs, plus useful negatives-in-domain
POS_ROOFS = {
 "roof_oblique_snow": ["snow covered rooftops seen from a taller building", "a view over snowy city roofs",
                       "a pitched metal roof covered with snow, seen from above"],
 "roof_oblique_clean": ["metal roofs of old city buildings seen from above, no snow", "a view over city rooftops in autumn"],
 "roof_on_snow": ["a photo taken standing on a snowy roof", "a worker clearing snow from a roof"],
 "roof_edge": ["snow overhang on the edge of a roof", "icicles hanging from a roof edge"],
}
NEG_ROOFS = NEG + ["a skyscraper", "a house in the countryside", "a cathedral dome", "snow covered cars",
                   "a snowy street seen from the ground", "a frozen river with a city skyline"]


def _labels(mode):
    pos, neg = (POS_ROOFS, NEG_ROOFS) if mode == "roofs" else (POS, NEG)
    return [(g, t) for g, ts in pos.items() for t in ts] + [("neg", t) for t in neg]


_TXT = {}


def _text(mode):
    if mode not in _TXT:
        lab = _labels(mode)
        with torch.no_grad():
            t = MODEL.encode_text(TOK([x for _, x in lab])); t /= t.norm(dim=-1, keepdim=True)
        _TXT[mode] = (lab, t)
    return _TXT[mode]


def score(paths, bs=32, mode="ground"):
    labels, TXT = _text(mode)
    out = []
    for i in range(0, len(paths), bs):
        ims = torch.stack([PRE(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        with torch.no_grad():
            f = MODEL.encode_image(ims); f /= f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ TXT.T).softmax(-1).numpy()
        for row in pr:
            g = {}
            for (grp, _), p in zip(labels, row):
                g[grp] = g.get(grp, 0) + p
            pos = 1 - g["neg"]
            best = max((k for k in g if k != "neg"), key=lambda k: g[k])
            out.append((float(pos), best, {k: round(float(v), 3) for k, v in g.items()}))
    return out
