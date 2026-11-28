//! Issue 1254 A1 — the fact corpus: a 16-relation factual-recall set with a
//! deterministic subject-wise 70/10/20 split.
//!
//! This is the PLATO'S KEY cell's data plane (see `.issues/1254_*.md` +
//! `.research/044_Platos_Key_CrossModel_Key_Transfer.md`): every fact is a
//! (relation, subject, object) triple rendered as a completion prompt
//! `"prefix + subject + suffix"` so BOTH models see the subject surface at a
//! token position the harness can compute exactly (the subject's last token),
//! which is where the donor key capture and the recipient REPLACE hook arm.
//!
//! Laws carried here:
//! - **Determinism (Issue 940 shape):** the split keys on blake3(subject) →
//!   SplitMix64 — pure arithmetic, no RNG state, identical across runs/boxes.
//!   Facts sharing a subject share a split (subject-wise: no subject leaks a
//!   split boundary).
//! - **Prompt integrity:** `encode(prompt) == encode(prefix) ++ encode(suffix)`
//!   is ASSERTED by the harness at run time for each tokenizer (BPE/SPM merge
//!   across a space boundary is stable; anything else refuses loud).
//! - **Matching honesty:** greedy continuations match on normalized
//!   word-boundary substring (accent-folded, punctuation→space, casefolded —
//!   except the two code-symbol relations, which match CASE-SENSITIVELY so
//!   "IN" (Indiana) never matches the word "in", and "Au" (gold) never
//!   matches prose).
//! - The corpus file itself is tracked (`i1254_a1_corpus.jsonl`) and pinned
//!   by digest; the audit bin prints the digest as the determinism receipt.

use serde_json::Value;

/// The tracked corpus (one JSON object per line:
/// `{"rel","subj","obj":[...]}`).
pub const CORPUS_JSONL: &str = include_str!("i1254_a1_corpus.jsonl");

/// A single fact. `obj` carries every acceptable surface (any-of matching);
/// the first entry is the canonical answer for reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub rel: &'static str,
    pub subj: String,
    pub obj: Vec<String>,
}

/// Prompt-template registry entry. The rendered prompt is
/// `capitalize_first(before + subj + after)` — capitalizing the first
/// ALPHABETICAL character lets article-less subjects ("Buckingham Palace is
/// located in") and article-carrying subjects ("the Eiffel Tower is located
/// in") share one code path.
pub struct Relation {
    pub code: &'static str,
    pub name: &'static str,
    pub before: &'static str,
    pub after: &'static str,
    /// Code-symbol relations match case-sensitively ("IN" vs "in", "Au" vs
    /// prose).
    pub case_sensitive: bool,
}

/// The 16 relations of record (the issue's pre-registered set).
pub const RELATIONS: &[Relation] = &[
    Relation {
        code: "c",
        name: "capital_country",
        before: "The capital of ",
        after: " is",
        case_sensitive: false,
    },
    Relation {
        code: "sc",
        name: "us_state_capital",
        before: "The capital of the U.S. state of ",
        after: " is",
        case_sensitive: false,
    },
    Relation {
        code: "sa",
        name: "us_state_abbrev",
        before: "The postal abbreviation for the U.S. state of ",
        after: " is",
        case_sensitive: true,
    },
    Relation {
        code: "cu",
        name: "currency_country",
        before: "The official currency of ",
        after: " is",
        case_sensitive: false,
    },
    Relation {
        code: "lg",
        name: "language_country",
        before: "The official language of ",
        after: " is",
        case_sensitive: false,
    },
    Relation {
        code: "ct",
        name: "city_country",
        before: "The city of ",
        after: " is located in",
        case_sensitive: false,
    },
    Relation {
        code: "co",
        name: "country_continent",
        before: "The country of ",
        after: " is in",
        case_sensitive: false,
    },
    Relation {
        code: "an",
        name: "author_novel",
        before: "The novel \"",
        after: "\" was written by",
        case_sensitive: false,
    },
    Relation {
        code: "pp",
        name: "painter_painting",
        before: "The painting \"",
        after: "\" was painted by",
        case_sensitive: false,
    },
    Relation {
        code: "cw",
        name: "composer_work",
        before: "The composition \"",
        after: "\" was composed by",
        case_sensitive: false,
    },
    Relation {
        code: "fc",
        name: "founder_company",
        before: "",
        after: " was founded by",
        case_sensitive: false,
    },
    Relation {
        code: "iv",
        name: "inventor_invention",
        before: "The ",
        after: " was invented by",
        case_sensitive: false,
    },
    Relation {
        code: "cs",
        name: "chemical_symbol",
        before: "The chemical symbol for ",
        after: " is",
        case_sensitive: true,
    },
    Relation {
        code: "lc",
        name: "landmark_city",
        before: "",
        after: " is located in",
        case_sensitive: false,
    },
    Relation {
        code: "cc",
        name: "character_creator",
        before: "The character ",
        after: " was created by",
        case_sensitive: false,
    },
    Relation {
        code: "pm",
        name: "planet_moon",
        before: "",
        after: " is a moon of",
        case_sensitive: false,
    },
];

/// Resolve a relation registry entry by code. Unknown codes refuse loud.
pub fn relation_of(code: &str) -> Result<&'static Relation, String> {
    RELATIONS
        .iter()
        .find(|r| r.code == code)
        .ok_or_else(|| format!("unknown relation code {code:?} (not in the 16-relation registry)"))
}

/// Render the completion prompt for a fact.
pub fn build_prompt(rel: &Relation, subj: &str) -> String {
    capitalize_first(&format!("{}{}{}", rel.before, subj, rel.after))
}

/// Split the prompt at the subject boundary: returns (prefix, suffix) with
/// `prefix + suffix == build_prompt(rel, subj)` byte-identically. The subject's
/// last token is `encode(prefix).len() - 1` (asserted equal on the full
/// encoding by the harness).
pub fn prompt_parts(rel: &Relation, subj: &str) -> (String, String) {
    let full = build_prompt(rel, subj);
    let prefix = capitalize_first(&format!("{}{}", rel.before, subj));
    debug_assert!(full.starts_with(&prefix));
    let suffix = full[prefix.len()..].to_string();
    (prefix, suffix)
}

fn capitalize_first(s: &str) -> String {
    let mut out = s.to_string();
    for (i, ch) in s.char_indices() {
        if ch.is_alphabetic() {
            let upper: String = ch.to_uppercase().collect();
            out.replace_range(i..i + ch.len_utf8(), &upper);
            return out;
        }
    }
    out
}

/// Parse the tracked corpus. Refusals name the line: unknown relation codes,
/// empty subjects/objects, duplicate (rel, subj) pairs, malformed JSON.
pub fn load_corpus() -> Result<Vec<Fact>, String> {
    let mut facts = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (idx, line) in CORPUS_JSONL.lines().enumerate() {
        fn bad(line_no: usize, why: impl std::fmt::Display) -> String {
            format!("corpus line {line_no}: {why}")
        }
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(line).map_err(|e| bad(idx + 1, format!("bad JSON: {e}")))?;
        let rel = v
            .get("rel")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(idx + 1, "missing rel"))?;
        let rel_static = RELATIONS
            .iter()
            .map(|r| r.code)
            .find(|c| *c == rel)
            .ok_or_else(|| bad(idx + 1, format!("unknown relation code {rel:?}")))?;
        let subj = v
            .get("subj")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(idx + 1, "missing subj"))?
            .to_string();
        let obj: Vec<String> = v
            .get("obj")
            .and_then(Value::as_array)
            .ok_or_else(|| bad(idx + 1, "missing obj array"))?
            .iter()
            .map(|o| {
                o.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad(idx + 1, "non-string obj entry"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if subj.trim().is_empty() {
            return Err(bad(idx + 1, "empty subject"));
        }
        if obj.is_empty() || obj.iter().any(|o| o.trim().is_empty()) {
            return Err(bad(idx + 1, "empty object"));
        }
        let key = (rel_static, subj.clone());
        if !seen.insert(key) {
            return Err(bad(
                idx + 1,
                format!("duplicate (rel, subj) = ({rel:?}, {subj:?})"),
            ));
        }
        facts.push(Fact {
            rel: rel_static,
            subj,
            obj,
        });
    }
    if facts.is_empty() {
        return Err("corpus parsed to zero facts".into());
    }
    Ok(facts)
}

/// The subject-wise split bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
    Test,
}

/// Subject-wise split: blake3(subject bytes) first 8 LE bytes → SplitMix64 →
/// `% 100` buckets. `<70` Train, `<80` Val, else Test. Same subject ⇒ same
/// bucket, forever, on every box.
pub fn split_of(subj: &str) -> Split {
    let digest = blake3::hash(subj.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[0..8]);
    split_bucket(u64::from_le_bytes(bytes))
}

fn split_bucket(mut z: u64) -> Split {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let z = z ^ (z >> 31);
    let pct = z % 100;
    if pct < 70 {
        Split::Train
    } else if pct < 80 {
        Split::Val
    } else {
        Split::Test
    }
}

/// BLAKE3 digest over the canonical split-assignment serialization
/// (`"subj\tsplit"` per fact, corpus order) — the determinism receipt the
/// audit bin prints and the tests pin.
pub fn split_assignment_digest(facts: &[Fact]) -> String {
    let mut h = blake3::Hasher::new();
    for f in facts {
        h.update(f.subj.as_bytes());
        h.update(b"\t");
        h.update(match split_of(&f.subj) {
            Split::Train => b"T",
            Split::Val => b"V",
            Split::Test => b"S",
        });
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
}

/// BLAKE3 over the raw corpus bytes — the data provenance receipt.
pub fn corpus_digest() -> String {
    blake3::hash(CORPUS_JSONL.as_bytes()).to_hex().to_string()
}

/// Fold Latin-1-class accented characters to their ASCII bases (é→e, ß→ss,
/// æ→ae, …). std has no NFD, so a fixed table it is — enough for every
/// surface this corpus or a model's continuation can produce.
fn fold_char(c: char, out: &mut String) {
    let folded: &str = match c {
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => "a",
        'é' | 'è' | 'ê' | 'ë' => "e",
        'í' | 'ì' | 'î' | 'ï' => "i",
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' => "o",
        'ú' | 'ù' | 'û' | 'ü' => "u",
        'ç' => "c",
        'ñ' => "n",
        'ý' | 'ÿ' => "y",
        'ß' => "ss",
        'æ' => "ae",
        'œ' => "oe",
        'ø' => "o",
        'ð' => "d",
        'ł' => "l",
        other => {
            out.push(other);
            return;
        }
    };
    out.push_str(folded);
}

/// Normalize a surface for matching: accent-fold, casefold (unless
/// case-sensitive), punctuation→space, collapse whitespace, trim.
pub fn normalize(s: &str, case_sensitive: bool) -> String {
    let mut folded = String::with_capacity(s.len());
    for c in s.chars() {
        let base = if case_sensitive { c } else { lower_char(c) };
        fold_char(base, &mut folded);
    }
    let mut out = String::with_capacity(folded.len());
    let mut last_space = true;
    let chars: Vec<char> = folded.chars().collect();
    for (i, c) in chars.iter().copied().enumerate() {
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if c == '.'
            && !last_space
            && chars
                .get(i + 1)
                .map(|n| !n.is_alphanumeric())
                .unwrap_or(true)
        {
            // Initialism terminal dot ("D.C." -> "dc"): a dot right after
            // an alphanumeric whose NEXT char is not alphanumeric folds
            // away without a space, so letter-dotted acronyms match their
            // plain spelling. Dots between alphanumerics ("3.5") stay
            // separators.
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim_end().to_string()
}

fn lower_char(c: char) -> char {
    let mut it = c.to_lowercase();
    let first = it.next().unwrap_or(c);
    // Multi-char lowercase expansions (ß→ss handled by fold_char; here any
    // 1:1 mapping suffices — take the first char, drop expansions).
    let _ = it.count();
    first
}

/// Does the greedy continuation contain any acceptable object as a
/// word-boundary substring? Both sides run through [`normalize`], then a
/// space-padded containment check approximates token boundaries without a
/// regex dependency.
pub fn answer_matches(generated: &str, obj: &[String], case_sensitive: bool) -> bool {
    let hay = format!(" {} ", normalize(generated, case_sensitive));
    obj.iter().any(|o| {
        let n = format!(" {} ", normalize(o, case_sensitive));
        !n.trim().is_empty() && hay.contains(&n)
    })
}

/// Number of generated tokens scanned for the answer (the continuation
/// window). One constant, pinned here, so reports and tests agree.
pub const MATCH_WINDOW_TOKENS: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_loads_and_is_well_formed() {
        let facts = load_corpus().expect("corpus must parse");
        assert!(facts.len() >= 1300, "corpus shrunk: {}", facts.len());
        let mut per_rel = std::collections::HashMap::new();
        for f in &facts {
            *per_rel.entry(f.rel).or_insert(0usize) += 1;
        }
        assert_eq!(per_rel.len(), RELATIONS.len(), "all 16 relations present");
        for r in RELATIONS {
            let n = per_rel.get(r.code).copied().unwrap_or(0);
            assert!(n >= 15, "relation {} too thin: {n}", r.name);
        }
    }

    #[test]
    fn prompts_render_and_split_at_the_subject() {
        let rel = relation_of("c").unwrap();
        let p = build_prompt(rel, "France");
        assert_eq!(p, "The capital of France is");
        let (pre, suf) = prompt_parts(rel, "France");
        assert_eq!(format!("{pre}{suf}"), p);
        assert!(pre.ends_with("France"));

        // Article-less subject under an empty `before` still capitalizes.
        let lc = relation_of("lc").unwrap();
        let p2 = build_prompt(lc, "Buckingham Palace");
        assert_eq!(p2, "Buckingham Palace is located in");
        // Article-carrying subject capitalizes its first letter.
        let p3 = build_prompt(lc, "the Eiffel Tower");
        assert_eq!(p3, "The Eiffel Tower is located in");
        let (pre3, suf3) = prompt_parts(lc, "the Eiffel Tower");
        assert_eq!(format!("{pre3}{suf3}"), p3);
        assert!(pre3.ends_with("Tower"));

        // Quoted-title relations keep the quotes outside the subject span.
        let an = relation_of("an").unwrap();
        let p4 = build_prompt(an, "Moby-Dick");
        assert_eq!(p4, "The novel \"Moby-Dick\" was written by");
        let (pre4, _) = prompt_parts(an, "Moby-Dick");
        assert!(!pre4.ends_with('"'));
        assert!(pre4.ends_with("Moby-Dick"));
    }

    #[test]
    fn split_is_deterministic_and_subject_wise() {
        // Same subject, same bucket — repeated calls agree.
        for subj in ["France", "the United States", "Io", "Sherlock Holmes"] {
            let a = split_of(subj);
            for _ in 0..4 {
                assert_eq!(a, split_of(subj), "split_of({subj}) is not pure");
            }
        }
        // The assignment digest is stable across calls.
        let facts = load_corpus().unwrap();
        let d1 = split_assignment_digest(&facts);
        let d2 = split_assignment_digest(&facts);
        assert_eq!(d1, d2);
    }

    #[test]
    fn split_distribution_is_balanced_on_the_corpus() {
        let facts = load_corpus().unwrap();
        let mut t = 0usize;
        let mut v = 0usize;
        let mut s = 0usize;
        for f in &facts {
            match split_of(&f.subj) {
                Split::Train => t += 1,
                Split::Val => v += 1,
                Split::Test => s += 1,
            }
        }
        let n = facts.len();
        // Loose balance bounds: the hash is uniform; these catch gross drift
        // (a broken bucket mapping would pin one bucket).
        assert!((t * 100) / n >= 60, "train share too low: {t}/{n}");
        assert!((t * 100) / n <= 80, "train share too high: {t}/{n}");
        assert!(v * 100 / n >= 4, "val share too low: {v}/{n}");
        assert!(s * 100 / n >= 10, "test share too low: {s}/{n}");
    }

    #[test]
    fn matching_is_word_boundary_and_accent_tolerant() {
        // Plain.
        assert!(answer_matches(
            " Paris. The capital",
            &["Paris".into()],
            false
        ));
        assert!(answer_matches(
            "the city of Paris, France",
            &["Paris".into()],
            false
        ));
        assert!(!answer_matches(
            "Parisian culture",
            &["Paris".into()],
            false
        ));
        assert!(!answer_matches(
            "the capital is Rome",
            &["Paris".into()],
            false
        ));
        // Articles + case.
        assert!(answer_matches(" The euro ", &["euro".into()], false));
        assert!(answer_matches("THE EURO", &["euro".into()], false));
        // Accents fold: Bogotá matches Bogota.
        assert!(answer_matches(
            " Bogotá, the capital",
            &["Bogota".into()],
            false
        ));
        assert!(answer_matches(
            " Les Misérables was written",
            &["Les Miserables".into()],
            false
        ));
        // Multi-token names.
        assert!(answer_matches(
            " Alexander Graham Bell in 1876.",
            &["Alexander Graham Bell".into()],
            false
        ));
        assert!(!answer_matches(
            " Alexander Graham",
            &["Alexander Graham Bell".into()],
            false
        ));
        // Punctuation inside the answer: a single spelling can't bridge
        // "D.C." vs "DC" — the corpus carries BOTH spellings in the obj
        // array (any-of matching), which is the design.
        assert!(answer_matches(
            " Washington, D.C. is",
            &["Washington DC".into(), "Washington, D.C.".into()],
            false
        ));
        assert!(answer_matches(
            " Washington DC is",
            &["Washington DC".into(), "Washington, D.C.".into()],
            false
        ));
    }

    #[test]
    fn code_symbol_matching_is_case_sensitive() {
        // "IN" (Indiana) must not match the word "in".
        assert!(answer_matches(" IN ", &["IN".into()], true));
        assert!(!answer_matches(
            " the state is in the US ",
            &["IN".into()],
            true
        ));
        // "Au" (gold) must not match prose.
        assert!(answer_matches(" Au ", &["Au".into()], true));
        assert!(!answer_matches(" a symbol ", &["Au".into()], true));
        assert!(answer_matches(" Na ", &["Na".into()], true));
    }

    #[test]
    fn known_split_buckets_are_pinned() {
        // Exact buckets pinned from a real run of i1254_corpus_audit (the
        // pins guard against blake3/finalizer drift). A change here is a
        // pre-registration change, never a test fix.
        assert_eq!(split_of("France"), Split::Test);
        assert_eq!(split_of("Io"), Split::Train);
        assert_eq!(split_of("Sherlock Holmes"), Split::Train);
    }
}
