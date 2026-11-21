"""Prompt utilities for Z-Image-Turbo.

Z-Image-Turbo runs at guidance_scale=0.0, so there is no negative prompt: writing "no X" puts X
into the conditioning and makes it MORE likely. These helpers rewrite negations into concrete
positive descriptions and expand terse prompts into the long photographic style the model prefers.
"""

import re

MIN_WORDS, MAX_WORDS = 80, 180

# Stem found in the negated phrase -> positive visual description that replaces it.
NEGATION_POSITIVES: list[tuple[tuple[str, ...], str]] = [
    (("people", "person", "crowd", "human", "bystander", "passerby"), "an empty, quiet setting"),
    (("text", "word", "letter", "watermark", "logo", "signature", "caption"), "clean unmarked surfaces"),
    (("clutter", "distraction", "busy background", "mess"), "a tidy, minimal environment"),
    (("background",), "a plain seamless backdrop"),
    (("makeup", "make-up", "cosmetic"), "a bare natural complexion"),
    (("blur", "blurry", "out of focus", "soft focus"), "tack-sharp focus"),
    (("shadow",), "soft, even illumination"),
    (("glasses", "sunglasses", "spectacles"), "a clear, unobstructed gaze"),
    (("jewelry", "jewellery", "accessor"), "minimal, unadorned styling"),
    (("smil",), "a calm, neutral expression"),
    (("hat", "cap", "hood"), "uncovered, natural hair"),
    (("tattoo",), "smooth, unmarked skin"),
    (("beard", "facial hair", "stubble"), "a clean-shaven face"),
    (("color", "colour"), "a black-and-white monochrome palette"),
    (("filter", "edit", "retouch", "airbrush"), "untouched, true-to-life rendering"),
    (("cartoon", "anime", "illustration", "painting", "drawing", "cgi", "3d render"), "a true photograph"),
]

_NEGATION = re.compile(
    r"\b(?:no|not|without|never|avoid(?:ing)?|free of|lacking|minus|"
    r"don'?t (?:show|include|add)|do not (?:show|include|add))\b\s+",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"([,;.\n]+)")

_HUMAN = re.compile(
    r"\b(?:woman|women|man|men|person|portrait|model|girl|guy|lady|gentleman|couple|face|athlete|dancer)\b",
    re.IGNORECASE,
)
_HAS = {
    "lighting": re.compile(r"\b(?:light|lighting|lit|sunlight|sun|sunset|sunrise|dawn|dusk|daylight|golden hour|blue hour|moonlight|candlelight|rim|softbox|neon|backlit|overcast)\b", re.I),
    "lens": re.compile(r"\b(?:\d{2,3}\s?mm|lens|depth of field|bokeh|aperture|f/\d)\b", re.I),
    "background": re.compile(r"\b(?:background|backdrop|setting|studio|environment)\b", re.I),
    "texture": re.compile(r"\b(?:texture|pores|detail|detailed)\b", re.I),
    "pose": re.compile(r"\b(?:pose|posing|standing|sitting|leaning|walking|lying|posture)\b", re.I),
}


def _positive_for(phrase: str) -> str:
    low = phrase.lower()
    for stems, positive in NEGATION_POSITIVES:
        if any(stem in low for stem in stems):
            return positive
    return ""  # unknown negation: drop it rather than mention the concept


def strip_negations(prompt: str) -> tuple[str, list[str]]:
    """Rewrite 'no X' / 'without X' clauses into positive descriptions.

    Returns (rewritten_prompt, list_of_human_readable_changes).
    """
    parts = _CLAUSE_SPLIT.split(prompt)
    out: list[str] = []
    added: list[str] = []
    changes: list[str] = []
    for part in parts:
        if _CLAUSE_SPLIT.fullmatch(part or " "):
            out.append(part)
            continue
        m = _NEGATION.search(part)
        if not m:
            out.append(part)
            continue
        before, negated = part[: m.start()], part[m.end():]
        before = re.sub(r"\b(?:and|but|with|or)\s*$", "", before.rstrip(), flags=re.I).rstrip()
        positive = _positive_for(negated)
        changes.append(f'"{m.group(0).strip()} {negated.strip()}" -> "{positive or "(removed)"}"')
        out.append(before)
        if positive and positive not in added:
            added.append(positive)

    text = "".join(out)
    text = re.sub(r"\s*([,;.])(\s*[,;.])+", r"\1", text)  # collapse punctuation left by removed clauses
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.")
    if added:
        text = f"{text}, {', '.join(added)}" if text else ", ".join(added)
    return text, changes


def scrub_terms(prompt: str) -> str:
    """The literal word 'camera' tends to render a camera in frame; describe the lens look instead."""
    text = re.sub(
        r"\b(?:shot|taken|captured|photographed)\s+(?:on|with|by)\s+(?:an?\s+)?(?:[\w-]+\s+){0,2}?cameras?\b",
        "85mm lens look", prompt, flags=re.IGNORECASE,
    )
    return re.sub(r"\bcameras?\b", "lens", text, flags=re.IGNORECASE)


def expand(prompt: str) -> str:
    """Rule-based expansion toward the model's preferred 80-180 word photographic description."""
    text = prompt.strip().rstrip(".")
    human = bool(_HUMAN.search(text))
    extras: list[str] = []
    if human and not _HAS["pose"].search(text):
        extras.append("relaxed, natural pose with confident posture and expressive hands")
    if not _HAS["lighting"].search(text):
        extras.append(
            "soft directional window light from the left with gentle falloff, a subtle warm rim light "
            "separating the subject from the background"
        )
    if not _HAS["lens"].search(text):
        extras.append("85mm lens look, shallow depth of field, crisp focus on the subject")
    if not _HAS["texture"].search(text):
        extras.append(
            "natural skin texture with fine pores and true-to-life tones"
            if human
            else "fine surface detail and true-to-life color"
        )
    if not _HAS["background"].search(text):
        extras.append("a clean, softly blurred background with muted tones")
    extras.append("balanced editorial composition, high dynamic range, realistic photograph")

    result = ". ".join([text] + extras) + "."
    words = result.split()
    if len(words) < MIN_WORDS:
        result += (
            " Subtle atmospheric depth, accurate anatomy and proportions, carefully observed materials "
            "and fabrics, natural color grading with gentle contrast, cinematic yet understated mood, "
            "every element rendered with clarity and intention."
        )
        words = result.split()
    if len(words) > MAX_WORDS:
        result = " ".join(words[:MAX_WORDS]).rstrip(",;") + "."
    return result


def rule_based_enhance(prompt: str) -> tuple[str, list[str]]:
    positive, changes = strip_negations(prompt)
    return expand(scrub_terms(positive)), changes


ENHANCER_SYSTEM_PROMPT = f"""You rewrite short image requests into prompts for a photorealistic text-to-image model.

Rules:
1. Output ONLY the rewritten prompt as a single paragraph of {MIN_WORDS}-{MAX_WORDS} words. No preamble, no quotes.
2. The model has NO negative prompt. Never use negations ("no", "not", "without", "avoid", "free of").
   Convert every exclusion into a concrete positive description (e.g. "no people" -> "an empty, quiet street").
3. Keep every concrete detail the user gave. Order: subject and appearance, pose and expression, wardrobe,
   setting, lighting setup, lens look (e.g. "85mm lens look, shallow depth of field"), texture, mood.
4. Describe natural skin texture and realistic materials. Prefer a clean, uncluttered background unless one is specified.
5. Never write the word "camera"; describe the lens look instead.
6. All people depicted are adults."""


# --- content guard ---------------------------------------------------------------------------
# Hard block on any reference to minors. Applied to image prompts and prompt enhancement.
_MINOR_TERMS = re.compile(
    r"\b(?:child|children|childlike|kid|kids|underage|under-age|preteen|pre-teen|tween|"
    r"teen|teens|teenage|teenager|schoolgirl|schoolboy|school uniform|loli|lolita|shota|"
    r"toddler|infant|juvenile|young-looking|barely legal)\b"
    r"|\b(?:[1-9]|1[0-7])\s*(?:yo|y/o|y\.o\.|years?[- ]old|year old)\b",
    re.IGNORECASE,
)


def blocked_reason(text: str) -> str | None:
    if _MINOR_TERMS.search(text):
        return "Prompt rejected: depictions must be clearly of adults."
    return None
