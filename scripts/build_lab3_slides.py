"""Write the slide numbers from ``results/lab3`` and compile the slides.

    python scripts/build_lab3_slides.py [--tectonic PATH]

No number is typed into ``slides/lab3_presentation.tex``: every one is a
macro of ``slides/lab3_numbers.tex``, generated here after the statements
check has passed. The fonts are the DejaVu files shipped with Matplotlib,
so the build needs no system fonts; their location depends on the machine,
hence ``slides/lab3_fonts.tex`` is generated and not stored in Git.
"""

import argparse
import math
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
from check_lab3_claims import AFTER, BEFORE, check_claims, sequence_path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"
TEX = SLIDES / "lab3_presentation.tex"
FIGURES = ROOT / "results/lab3"


SUBSCRIPTS = dict(zip("₀₁₂₃₄₋ₘ", "01234-m"))


def _tex(text):
    """{M, R₂} of the summary as TeX: \\{M, R_{2}\\}."""
    out, index = [], 0
    while index < len(text):
        run = ""
        while index < len(text) and text[index] in SUBSCRIPTS:
            run += SUBSCRIPTS[text[index]]
            index += 1
        if run:
            out.append(f"_{{{run}}}")
            continue
        out.append({"{": r"\{", "}": r"\}"}.get(text[index], text[index]))
        index += 1
    return "".join(out)


def _bound(value, digits):
    """Round up: the slides quote this number as "not more than"."""
    scale = 10**digits
    return f"{math.ceil(value * scale) / scale:.{digits}f}"


def numbers(summary, path):
    """Macro name -> text, for everything the slides quote."""
    config, problems = summary["config"], summary["problems"]
    profile = {case["problem"]: case for case in summary["stability_profile"]}
    alpha = profile["rlc"]["levels"].index(config["intervention_alpha"])
    truth_ratio = max(
        max(row[:-1]) for case in profile.values() for row in case["ratio"]
    )
    rlc = problems["rlc"]
    distance = max(
        value
        for stage in rlc["stages"].values()
        for value in stage["total_variation"].values()
    )
    margin = min(
        problem["tolerance"] / problem["validation_rmse"]
        for problem in problems.values()
    )
    values = {
        "Repeats": config["intervention_repeats"],
        "Alpha": f"{config['intervention_alpha']:g}",
        "Weight": f"{config['stability_weight']:g}",
        "SwapShare": f"{config['swap_probability']:g}",
        "RhoTruthMax": _bound(truth_ratio, 2),
        "RlcGraphs": 2 ** len(rlc["edges"]),
        "RlcDistance": _bound(distance, 3),
        "LostStep": path["lost"],
        "BackStep": path["back"],
        "TruthStep": path["truth"],
        "RmseMargin": f"{int(margin)}",
    }
    for stage, suffix in ((BEFORE, "Before"), (AFTER, "After")):
        for name, problem in problems.items():
            share = problem["stages"][stage]["selected_share"]
            values[f"{name.capitalize()}Share{suffix}"] = f"{share:.3f}"
    # The graphs of the stability figure are the truth plus one extra edge,
    # listed last.
    for name in ("rlc", "silica"):
        prefix = f"RhoExtra{name.capitalize()}"
        values[prefix] = f"{profile[name]['ratio'][alpha][-1]:.2f}"
        values[f"{prefix}Edge"] = _tex(profile[name]["edges"][-1])
    for name, problem in problems.items():
        values[f"{name.capitalize()}Selected"] = _tex(
            problem["stages"][AFTER]["selected"]
        )
    return values


def check_macros(values):
    """A number that no slide quotes is a leftover of a removed slide."""
    tex = TEX.read_text(encoding="utf-8")
    unused = [
        name for name in values if not re.search(rf"\\{name}(?![A-Za-z])", tex)
    ]
    if unused:
        raise SystemExit(
            "Numbers not quoted on the slides: " + ", ".join(unused)
        )


def write_numbers(values):
    lines = ["% Written by scripts/build_lab3_slides.py; do not edit."]
    for name, value in values.items():
        if not name.isalpha():
            raise ValueError(f"TeX macro names are letters only: {name!r}")
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    (SLIDES / "lab3_numbers.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_fonts():
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    path = f"Path={fonts}/, Extension=.ttf"
    families = {
        "DejaVuSans": (
            ("setsansfont", r"newfontfamily\cyrillicfontsf"),
            "BoldFont=DejaVuSans-Bold, ItalicFont=DejaVuSans-Oblique, "
            "BoldItalicFont=DejaVuSans-BoldOblique, Scale=0.88",
        ),
        "DejaVuSerif": (
            ("setmainfont", r"newfontfamily\cyrillicfont"),
            "BoldFont=DejaVuSerif-Bold, ItalicFont=DejaVuSerif-Italic, "
            "BoldItalicFont=DejaVuSerif-BoldItalic, Scale=0.88",
        ),
        "DejaVuSansMono": (
            ("setmonofont", r"newfontfamily\cyrillicfonttt"),
            "BoldFont=DejaVuSansMono-Bold, "
            "ItalicFont=DejaVuSansMono-Oblique, Scale=0.8",
        ),
    }
    lines = ["% Written by scripts/build_lab3_slides.py; machine-specific."]
    for family, (commands, shapes) in families.items():
        for command in commands:
            lines.append(rf"\{command}{{{family}}}[{path}, {shapes}]")
    (SLIDES / "lab3_fonts.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def check_figures():
    """Every included figure must be an output of the executed notebook."""
    names = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        TEX.read_text(encoding="utf-8"),
    )
    missing = [
        name for name in names if not (FIGURES / f"{name}.png").exists()
    ]
    if missing:
        raise SystemExit(
            "Run scripts/execute_lab3.py first; missing figures: "
            + ", ".join(missing)
        )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", default=shutil.which("tectonic"))
    arguments = parser.parse_args()
    if arguments.tectonic is None:
        raise SystemExit(
            "tectonic not found; install it (conda install -c conda-forge "
            "tectonic) or pass --tectonic PATH"
        )
    summary = check_claims(ROOT)
    names = check_figures()
    values = numbers(summary, sequence_path(ROOT))
    check_macros(values)
    write_numbers(values)
    write_fonts()
    subprocess.run(
        [arguments.tectonic, "-X", "compile", TEX.name], cwd=SLIDES, check=True
    )
    print(f"Wrote {TEX.with_suffix('.pdf')} with {len(names)} figures")


if __name__ == "__main__":
    main()
