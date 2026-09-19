"""Tests for InverPINN's initial, implementation-free repository scaffold."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_package_components_have_separate_locations() -> None:
    """Reserve independent locations for each future research concern.

    Separation prevents an eventual neural-network implementation from hiding
    the advection--diffusion mathematics in model code.
    """
    package = REPOSITORY_ROOT / "src" / "inverpinn"
    expected_directories = (
        "data",
        "models",
        "physics",
        "training",
        "evaluation",
        "visualization",
    )

    assert all((package / directory).is_dir() for directory in expected_directories)


def test_requested_top_level_directories_exist() -> None:
    """Keep data, work products, and tests in predictable locations."""
    expected_directories = (
        "configs",
        "data/raw",
        "data/processed",
        "data/synthetic",
        "scripts",
        "notebooks",
        "experiments",
        "tests",
        "results",
    )

    assert all((REPOSITORY_ROOT / directory).is_dir() for directory in expected_directories)


def test_named_configuration_placeholders_exist() -> None:
    """Ensure each planned research stage has a version-controlled config file."""
    expected_configs = ("synthetic.yaml", "pinn.yaml", "almaty.yaml")

    assert all((REPOSITORY_ROOT / "configs" / config).is_file() for config in expected_configs)


def test_declared_dependencies_include_requested_research_stack() -> None:
    """Verify the dependency manifest names the requested Python libraries."""
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text().lower()
    expected_packages = (
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "xarray",
        "netcdf4",
        "scikit-learn",
        "pyyaml",
        "pytest",
    )

    assert all(package in requirements for package in expected_packages)
