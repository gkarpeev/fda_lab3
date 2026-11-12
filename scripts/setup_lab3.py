"""Install the laboratory 3 Jupyter kernel inside the virtual environment."""

from pathlib import Path

from ipykernel.kernelspec import install

from emlab.lab3_config import KERNEL_DISPLAY_NAME, KERNEL_NAME

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = install(
        prefix=str(ROOT / ".venv"),
        kernel_name=KERNEL_NAME,
        display_name=KERNEL_DISPLAY_NAME,
    )
    print(f"Installed {KERNEL_NAME}: {destination}")


if __name__ == "__main__":
    main()
