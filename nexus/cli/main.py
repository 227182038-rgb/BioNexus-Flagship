"""BioNexus Command Line Interface."""

from __future__ import annotations

import typer

from nexus.application.app import BioNexus


app = typer.Typer(
    help="BioNexus Scientific Research Platform"
)

bionexus = BioNexus()


@app.command()
def analyze(
    query: str,
):
    """Analyze a biomedical query."""

    report = bionexus.analyze(query)

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(" BioNexus Scientific Analysis")
    typer.echo("=" * 60)

    typer.echo(f"Query: {report.query}")

    if report.metadata:
        typer.echo("")

        typer.echo("Subject")
        typer.echo("-" * 20)

        if report.metadata.get("subject_name"):
            typer.echo(
                f"Name: {report.metadata['subject_name']}"
            )

        if report.metadata.get("subject_id"):
            typer.echo(
                f"ID: {report.metadata['subject_id']}"
            )

        if report.metadata.get("subject_type"):
            typer.echo(
                f"Type: {report.metadata['subject_type']}"
            )

        if report.metadata.get("evidence_count"):
            typer.echo(
                f"Evidence: "
                f"{report.metadata['evidence_count']}"
            )

    if report.explanation is not None:
        typer.echo("")

        typer.echo("Scientific Summary")
        typer.echo("-" * 20)

        if report.explanation.summary:
            typer.echo(
                report.explanation.summary
            )

        if report.explanation.reasoning:
            typer.echo("")

            for point in report.explanation.reasoning:
                typer.echo(
                    f"- {point}"
                )

        if report.explanation.evidence:
            typer.echo("")

            typer.echo("Evidence")
            typer.echo("-" * 20)

            for item in report.explanation.evidence:
                typer.echo(
                    f"- {item}"
                )

        typer.echo("")

        typer.echo(
            f"Representation confidence: "
            f"{report.explanation.confidence:.3f}"
        )

        if report.scientific_confidence is not None:
            sc = report.scientific_confidence

            typer.echo(
                f"Scientific conclusion confidence: "
                f"{sc.overall:.3f}"
            )

            typer.echo(
                f"  Evidence quality: "
                f"{sc.evidence_quality:.3f}"
            )

            typer.echo(
                f"  Source diversity: "
                f"{sc.source_diversity:.3f}"
            )

            typer.echo(
                f"  Relationship support: "
                f"{sc.relationship_support:.3f}"
            )

            typer.echo(
                f"  Literature support: "
                f"{sc.literature_support:.3f}"
            )

            typer.echo(
                f"  Consistency: "
                f"{sc.consistency:.3f}"
            )

    typer.echo("")
    typer.echo("Ranked Entities")
    typer.echo("-" * 20)

    for index, entity in enumerate(
        report.ranked_entities,
        start=1,
    ):
        typer.echo(
            f"{index}. "
            f"{entity.entity_id} "
            f"({entity.score:.3f})"
        )

    typer.echo("")
    typer.echo("Hypotheses")
    typer.echo("-" * 20)

    if report.hypotheses:
        for hypothesis in report.hypotheses:
            typer.echo(
                f"- {hypothesis.title}: "
                f"{hypothesis.description}"
            )
    else:
        typer.echo(
            "- No evidence-backed hypotheses generated."
        )

    typer.echo("")
    typer.echo("Contradictions")
    typer.echo("-" * 20)

    if report.contradictions:
        for contradiction in report.contradictions:
            typer.echo(
                f"- {contradiction.topic}: "
                f"{contradiction.claim_a} "
                f"vs "
                f"{contradiction.claim_b}"
            )
    else:
        typer.echo(
            "- No contradictions detected."
        )


if __name__ == "__main__":
    app()

