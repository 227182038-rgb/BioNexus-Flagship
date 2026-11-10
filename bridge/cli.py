"""BioNexus unified CLI — one entry point for BioKit, NEXUS, and the bridge.

Commands
--------
- ``bionexus biokit <subcommand>`` — BioKit 2.0 commands (gc, orfs, translate, version)
- ``bionexus nexus <subcommand>`` — NEXUS commands (interpret, validate, report)
- ``bionexus run <program> --inputs <json>`` — run a BioKit program via the bridge
- ``bionexus programs`` — list all registered BioKit programs
- ``bionexus version`` — print the BioNexus platform version

Typer's ``@app.command()`` decorator erases function types at the mypy level.
To preserve strict typing we define each command body as a typed function and
register them via ``app.command()(fn)`` rather than using decorator syntax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any


def _no_typer_option(*args: Any, **kwargs: Any) -> None:
    """Stub for :func:`typer.Option` when typer is not installed."""
    return None


typer: Any = None
Option: Any = _no_typer_option
try:
    import typer as _typer_mod
    from typer import Option as _typer_option

    typer = _typer_mod
    Option = _typer_option
except ImportError:  # pragma: no cover
    pass


if typer is not None:
    app: Any = typer.Typer(
        name="bionexus",
        help="BioNexus — BioKit computes. Nexus understands.",
        no_args_is_help=True,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )
    biokit_app: Any = typer.Typer(help="BioKit 2.0 commands")
    nexus_app: Any = typer.Typer(help="NEXUS commands")
else:
    app = None
    biokit_app = None
    nexus_app = None


# ---------------------------------------------------------------------------
# Command implementations (typed, no decorator)
# ---------------------------------------------------------------------------


def _version() -> None:
    """Print the BioNexus platform version."""
    from rich.console import Console

    from bridge import __version__ as bridge_version

    Console().print(f"BioNexus Platform [bold]{bridge_version}[/bold]")


def _programs() -> None:
    """List all registered BioKit programs available through the bridge."""
    from rich.console import Console
    from rich.table import Table

    from bridge.biokit_programs import BioKitProgramRegistry

    console = Console()
    table = Table(title="Registered BioKit Programs")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    for name in BioKitProgramRegistry.list_programs():
        cls = BioKitProgramRegistry.get(name)
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        table.add_row(name, doc)
    console.print(table)


def _run(
    program: Annotated[str, Any],  # typer.Argument at registration time
    inputs: Annotated[str, Any] = "{}",
) -> None:
    """Run a BioKit program through the bridge and print the JSON output."""
    from rich.console import Console

    from bridge.biokit_programs import BioKitProgramRegistry
    from bridge.prelude import quickstart

    console = Console()
    if inputs.startswith("@"):
        inputs = Path(inputs[1:]).read_text()
    try:
        inputs_dict: dict[str, Any] = json.loads(inputs)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON inputs:[/red] {exc}")
        raise SystemExit(1) from exc

    nx = quickstart()
    try:
        output = nx.run_biokit(program, inputs_dict)
    except KeyError as exc:
        console.print(f"[red]Unknown program:[/red] {program}")
        console.print(f"Available: {', '.join(BioKitProgramRegistry.list_programs())}")
        raise SystemExit(1) from exc

    console.print_json(json.dumps(output.outputs, default=str))


def _biokit_gc(
    sequence: Annotated[str, Any],
) -> None:
    """Compute GC content of a sequence."""
    from biokit.statistics.sequence_stats import gc_fraction
    from rich.console import Console

    Console().print(f"[bold]Total GC:[/bold] {gc_fraction(sequence) * 100:.2f}%")


def _biokit_orfs(
    sequence: Annotated[str, Any],
    min_length: Annotated[int, Any] = 30,
) -> None:
    """Find ORFs in a sequence."""
    from biokit.orf import ORFFinder
    from rich.console import Console
    from rich.table import Table

    results = ORFFinder(minimum_length=min_length).find(sequence)
    console = Console()
    table = Table(title="ORFs")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Frame", justify="right")
    table.add_column("Strand")
    table.add_column("Protein")
    for orf in results:
        table.add_row(
            str(orf.start), str(orf.end), str(orf.frame), orf.strand, orf.protein_sequence
        )
    console.print(table)


def _biokit_translate(
    sequence: Annotated[str, Any],
    frame: Annotated[int, Any] = 1,
) -> None:
    """Translate a DNA sequence to a protein."""
    from biokit.translation import Translator
    from rich.console import Console

    protein = Translator().translate(sequence, frame=frame)
    Console().print(f"[bold]Protein:[/bold] {protein}")


def _biokit_version() -> None:
    """Print the BioKit version."""
    from biokit import __version__ as biokit_v
    from rich.console import Console

    Console().print(f"BioKit [bold]{biokit_v}[/bold]")


def _nexus_interpret(
    question: Annotated[str, Any],
    agent: Annotated[str, Any] = "literature",
    biokit_output: Annotated[str | None, Any] = None,
) -> None:
    """Ask NEXUS to interpret a question (optionally grounded in BioKit output)."""
    from rich.console import Console

    from bridge.prelude import quickstart

    console = Console()
    nx = quickstart()
    context: dict[str, Any] | None = None
    if biokit_output:
        context = {"biokit_output": json.loads(Path(biokit_output).read_text())}
    result = nx.interpret(question, agent=agent, context=context)
    console.print(f"[bold]Answer:[/bold] {result.answer}")
    console.print(f"[bold]Confidence:[/bold] {result.confidence.value:.2%}")


def _nexus_agents() -> None:
    """List the registered NEXUS agents."""
    from rich.console import Console
    from rich.table import Table

    from bridge.prelude import quickstart

    nx = quickstart()
    console = Console()
    table = Table(title="NEXUS Agents")
    table.add_column("Name", style="cyan")
    for name in nx.list_agents():
        table.add_row(name)
    console.print(table)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register_commands() -> None:
    """Register every command implementation with the Typer apps."""
    if typer is None or app is None:
        return
    # Top-level
    app.command("version")(_version)
    app.command("programs")(_programs)
    app.command("run")(_run)
    # BioKit sub-app
    biokit_app.command("gc")(_biokit_gc)
    biokit_app.command("orfs")(_biokit_orfs)
    biokit_app.command("translate")(_biokit_translate)
    biokit_app.command("version")(_biokit_version)
    # NEXUS sub-app
    nexus_app.command("interpret")(_nexus_interpret)
    nexus_app.command("agents")(_nexus_agents)
    # Wire sub-apps
    app.add_typer(biokit_app, name="biokit")
    app.add_typer(nexus_app, name="nexus")


_register_commands()


def main() -> None:
    """Entry point for ``python -m bridge``."""
    if typer is None:
        print("Error: typer is not installed. Run `pip install typer rich`.")
        raise SystemExit(1)
    app()


if __name__ == "__main__":
    main()
