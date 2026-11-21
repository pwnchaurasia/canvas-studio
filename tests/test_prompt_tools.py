import prompt_tools as pt


def test_negations_become_positive():
    out, changes = pt.strip_negations("a quiet street at dawn, no people, without text")
    low = out.lower()
    assert "people" not in low and "text" not in low
    assert "empty, quiet setting" in low and "clean unmarked surfaces" in low
    assert len(changes) == 2


def test_unknown_negation_dropped():
    out, _ = pt.strip_negations("a red vase on a table, no zebras")
    assert "zebra" not in out.lower() and out.startswith("a red vase")


def test_no_negation_untouched():
    assert pt.strip_negations("a red vase on a table")[0] == "a red vase on a table"


def test_expand_word_range_and_terms():
    text, _ = pt.rule_based_enhance("portrait of a woman, shot on a camera, no makeup")
    words = len(text.split())
    assert pt.MIN_WORDS <= words <= pt.MAX_WORDS
    low = text.lower()
    assert "camera" not in low
    assert "85mm lens look" in low
    assert "natural skin texture" in low
    assert "bare natural complexion" in low
    for neg in (" no ", " without ", " not "):
        assert neg not in f" {low} "


def test_guard():
    assert pt.blocked_reason("a 17 yo model")
    assert pt.blocked_reason("schoolgirl outfit")
    assert pt.blocked_reason("portrait of a 30 year old woman") is None
    assert pt.blocked_reason("a kidney-shaped pool") is None
