"""Static check of the two papers, for machines with no LaTeX: every \\ref has a \\label, every \\cite a \\bibitem,
every figure file exists, braces and \\begin/\\end balance. Not a compile. Run from v1/: python paper/lint.py"""
import re, sys
from pathlib import Path

bad = 0
for f in sorted(Path(__file__).parent.glob("main*.tex")):
    s = re.sub(r"(?<!\\)%.*", "", f.read_text(encoding="utf-8"))
    labels, bibs = set(re.findall(r"\\label\{([^}]+)\}", s)), set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    cites = {c.strip() for g in re.findall(r"\\cite\{([^}]+)\}", s) for c in g.split(",")}
    figs = [g for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)
            if not any((f.parent / (g + e)).exists() for e in ("", ".pdf", ".png"))]
    b, e = re.findall(r"\\begin\{(\w+\*?)\}", s), re.findall(r"\\end\{(\w+\*?)\}", s)
    plain = s.replace("\\{", "").replace("\\}", "")
    errs = {"ref without label": set(re.findall(r"\\ref\{([^}]+)\}", s)) - labels, "cite without bibitem": cites - bibs,
            "bibitem never cited": bibs - cites, "missing figure": figs,
            "begin/end": [x for x in set(b + e) if b.count(x) != e.count(x)],
            "braces": [] if plain.count("{") == plain.count("}") else [plain.count("{") - plain.count("}")]}
    print(f.name, "todos left:", len(re.findall(r"\\todo\{", s)))
    for k, v in errs.items():
        if v:
            bad += 1
            print("  ", k, sorted(map(str, v)))
sys.exit(bad > 0)
