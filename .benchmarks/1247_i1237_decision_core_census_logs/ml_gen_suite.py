"""Issue 1237 T-ml: a multilingual decision suite rendered from the tetris v4
fixture's closed templated vocabulary (es / th / ja phrase tables).

Every option sentence parses as
  "The piece <holes> under it <pos>, <surface>, and the stack <height>[, and clears N line(s)]."
and every state sentence is a fixed set of clauses. Unparseable text raises (no
silent passthrough). Option order, state ids and features are kept; the
records are interleaved round-robin across languages, so both census halves
(fit / eval) see every language.

usage: python3 -I gen_suite.py <tetris_v4.jsonl> <out.jsonl> <question_key> [langs]
"""
import json
import re
import sys

LANGS = ["es", "th", "ja"]

QUESTIONS = {
    "clean": {
        "es": "¿La pila se ve limpia?",
        "th": "กองบล็อกดูเรียบร้อยไหม",
        "ja": "スタックはきれいに見えますか？",
    },
    "holes": {
        "es": "¿La pieza deja huecos debajo?",
        "th": "ชิ้นส่วนนี้ทิ้งช่องว่างไว้ข้างใต้หรือไม่",
        "ja": "ピースは下に穴を残しますか？",
    },
}

HOLES = {
    "leaves no holes": {"es": "no deja huecos", "th": "ไม่ทิ้งช่องว่าง", "ja": "下に穴を残さず"},
    "leaves one hole": {"es": "deja un hueco", "th": "ทิ้งช่องว่างหนึ่งช่อง", "ja": "下に穴を一つ残し"},
    "leaves two holes": {"es": "deja dos huecos", "th": "ทิ้งช่องว่างสองช่อง", "ja": "下に穴を二つ残し"},
    "leaves a few holes": {"es": "deja algunos huecos", "th": "ทิ้งช่องว่างไม่กี่ช่อง", "ja": "下に穴をいくつか残し"},
    "leaves many holes": {"es": "deja muchos huecos", "th": "ทิ้งช่องว่างหลายช่อง", "ja": "下に穴をたくさん残し"},
}
POS = {
    "on the left edge": {"es": "en el borde izquierdo", "th": "ที่ขอบซ้าย", "ja": "左端で"},
    "on the right edge": {"es": "en el borde derecho", "th": "ที่ขอบขวา", "ja": "右端で"},
    "on the left side": {"es": "en el lado izquierdo", "th": "ทางด้านซ้าย", "ja": "左側で"},
    "on the right side": {"es": "en el lado derecho", "th": "ทางด้านขวา", "ja": "右側で"},
    "in the middle": {"es": "en el centro", "th": "ตรงกลาง", "ja": "中央で"},
}
SURF = {
    "fills a deep gap": {"es": "rellena un hueco profundo", "th": "เติมร่องลึก", "ja": "深い溝を埋め"},
    "sits flat on the surface": {"es": "queda plana sobre la superficie", "th": "วางราบบนพื้นผิว", "ja": "表面に平らに置かれ"},
    "makes a small bump on top": {"es": "forma un pequeño bulto arriba", "th": "ทำให้เกิดปุ่มเล็กด้านบน", "ja": "上に小さな出っ張りを作り"},
    "makes a tall step on top": {"es": "forma un escalón alto arriba", "th": "ทำให้เกิดขั้นสูงด้านบน", "ja": "上に高い段差を作り"},
}
HEIGHT = {
    "stays low": {"es": "se mantiene baja", "th": "กองยังต่ำอยู่", "ja": "スタックは低いままです"},
    "stands medium": {"es": "queda a media altura", "th": "กองสูงปานกลาง", "ja": "スタックは中くらいの高さです"},
    "grows tall": {"es": "crece alta", "th": "กองสูงขึ้นมาก", "ja": "スタックは高くなります"},
}
CLEARS = {
    "a line": {"es": ", y despeja una línea", "th": " และเคลียร์ได้หนึ่งแถว", "ja": "。一列消します"},
    "two lines": {"es": ", y despeja dos líneas", "th": " และเคลียร์ได้สองแถว", "ja": "。二列消します"},
    "three lines": {"es": ", y despeja tres líneas", "th": " และเคลียร์ได้สามแถว", "ja": "。三列消します"},
    "four lines": {"es": ", y despeja cuatro líneas", "th": " และเคลียร์ได้สี่แถว", "ja": "。四列消します"},
}
OPT = re.compile(r"^The piece (.+?) under it (.+?), (.+?), and the stack (stays low|stands medium|grows tall)(?:, and clears (.+))?\.$")


def option(sentence, lang):
    m = OPT.match(sentence)
    if not m:
        raise ValueError(f"unparsed option: {sentence!r}")
    h, p, s, ht, cl = m.groups()
    clear = CLEARS[cl][lang] if cl else ""
    H, P, S, T = HOLES[h][lang], POS[p][lang], SURF[s][lang], HEIGHT[ht][lang]
    if lang == "es":
        return f"La pieza {H} debajo {P}, {S} y la pila {T}{clear}."
    if lang == "th":
        return f"ชิ้นส่วน{H}ไว้ข้างใต้{P} {S} และ{T}{clear}"
    return f"ピースは{P}{H}、{S}、{T}{clear}。"


PIECE = {
    "the S shaped piece": {"es": "la pieza en forma de S", "th": "ชิ้นรูปตัว S", "ja": "S字型のピース"},
    "the T shaped piece": {"es": "la pieza en forma de T", "th": "ชิ้นรูปตัว T", "ja": "T字型のピース"},
    "the Z shaped piece": {"es": "la pieza en forma de Z", "th": "ชิ้นรูปตัว Z", "ja": "Z字型のピース"},
    "the left leaning ell piece": {"es": "la ele inclinada a la izquierda", "th": "ชิ้นตัวแอลเอียงซ้าย", "ja": "左向きのL字ピース"},
    "the right leaning ell piece": {"es": "la ele inclinada a la derecha", "th": "ชิ้นตัวแอลเอียงขวา", "ja": "右向きのL字ピース"},
    "the long straight piece": {"es": "la pieza larga y recta", "th": "ชิ้นยาวตรง", "ja": "長いまっすぐなピース"},
    "the square piece": {"es": "la pieza cuadrada", "th": "ชิ้นสี่เหลี่ยม", "ja": "四角いピース"},
}
WHERE = {
    "left": {"es": "la izquierda", "th": "ด้านซ้าย", "ja": "左"},
    "right": {"es": "la derecha", "th": "ด้านขวา", "ja": "右"},
    "middle": {"es": "el centro", "th": "ตรงกลาง", "ja": "中央"},
}
STATE_HOLES = {
    "There are no holes under the blocks": {"es": "No hay huecos bajo los bloques.", "th": "ไม่มีช่องว่างใต้บล็อก", "ja": "ブロックの下に穴はありません。"},
    "There is one hole under the blocks": {"es": "Hay un hueco bajo los bloques.", "th": "มีช่องว่างหนึ่งช่องใต้บล็อก", "ja": "ブロックの下に穴が一つあります。"},
    "There are two holes under the blocks": {"es": "Hay dos huecos bajo los bloques.", "th": "มีช่องว่างสองช่องใต้บล็อก", "ja": "ブロックの下に穴が二つあります。"},
    "There are a few holes under the blocks": {"es": "Hay algunos huecos bajo los bloques.", "th": "มีช่องว่างไม่กี่ช่องใต้บล็อก", "ja": "ブロックの下に穴がいくつかあります。"},
}
STACK_FLAT = re.compile(r"^The stack stands (low|of medium height) and the surface is mostly flat$")
STACK_TILT = re.compile(r"^The stack stands (of medium height|tall), tall on the (left|right|middle) and low on the (left|right|middle)$")
LEVEL = {
    "low": {"es": "baja", "th": "ต่ำ", "ja": "低く"},
    "of medium height": {"es": "a media altura", "th": "สูงปานกลาง", "ja": "中くらいの高さで"},
    "tall": {"es": "alta", "th": "สูง", "ja": "高く"},
}


def state_clause(c, lang):
    if c in STATE_HOLES:
        return STATE_HOLES[c][lang]
    m = STACK_FLAT.match(c)
    if m:
        lv = LEVEL[m.group(1)][lang]
        return {"es": f"La pila está {lv} y la superficie es casi plana.",
                "th": f"กองบล็อก{lv}และพื้นผิวค่อนข้างเรียบ",
                "ja": f"スタックは{lv}、表面はほぼ平らです。"}[lang]
    m = STACK_TILT.match(c)
    if m:
        lv, a, b = LEVEL[m.group(1)][lang], WHERE[m.group(2)][lang], WHERE[m.group(3)][lang]
        return {"es": f"La pila está {lv}, alta a {a} y baja a {b}.",
                "th": f"กองบล็อก{lv} สูงที่{a}และต่ำที่{b}",
                "ja": f"スタックは{lv}、{a}が高く{b}が低いです。"}[lang]
    m = re.match(r"^The next piece is (.+)$", c)
    if m:
        p = PIECE[m.group(1)][lang]
        return {"es": f"La siguiente pieza es {p}.", "th": f"ชิ้นถัดไปคือ{p}", "ja": f"次のピースは{p}です。"}[lang]
    m = re.match(r"^(The .+) is falling$", c)
    if m:
        p = PIECE["t" + m.group(1)[1:]][lang]
        return {"es": f"Está cayendo {p}.", "th": f"{p}กำลังตกลงมา", "ja": f"{p}が落ちています。"}[lang]
    raise ValueError(f"unparsed state clause: {c!r}")


def state(sentence, lang):
    parts = [state_clause(c, lang) for c in sentence.rstrip(".").split(". ")]
    return (" " if lang in ("es", "th") else "").join(parts)


def main():
    src, out, qkey = sys.argv[1], sys.argv[2], sys.argv[3]
    langs = sys.argv[4].split(",") if len(sys.argv) > 4 else LANGS
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    rows = [r for r in rows if r.get("state_id") != "_meta"]
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            lang = langs[i % len(langs)]
            rec = {
                "state_id": f"{r['state_id']}|lang:{lang}",
                "lang": lang,
                "question": QUESTIONS[qkey][lang],
                "state_sentence": state(r["state_sentence"], lang),
                "options": [{"sentence": option(o["sentence"], lang), "features": o["features"]} for o in r["options"]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records -> {out}")


if __name__ == "__main__":
    main()
