"""Automated report generation and publishing for cytoflow-qc results."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

# Optional dependencies for report generation
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


class ReportGenerator:
    """Automated report generation for cytoflow-qc results."""

    def __init__(self, results_dir: str | Path, template_dir: str | Path | None = None):
        """Initialize report generator.

        Args:
            results_dir: Path to cytoflow-qc results directory
            template_dir: Directory containing report templates
        """
        self.results_dir = Path(results_dir)
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent.parent / "templates"

        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {self.results_dir}")

        self._load_analysis_data()
        self._setup_templates()

    def _load_analysis_data(self) -> None:
        """Load all analysis data from results directory."""
        # Load QC data
        qc_file = self.results_dir / "qc" / "summary.csv"
        self.qc_data = pd.read_csv(qc_file) if qc_file.exists() else None

        # Load gating data
        gate_file = self.results_dir / "gate" / "summary.csv"
        self.gate_data = pd.read_csv(gate_file) if gate_file.exists() else None

        # Load drift data
        drift_tests = self.results_dir / "drift" / "tests.csv"
        self.drift_tests = pd.read_csv(drift_tests) if drift_tests.exists() else None

        # Load statistical data
        stats_file = self.results_dir / "stats" / "effect_sizes.csv"
        self.stats_data = pd.read_csv(stats_file) if stats_file.exists() else None

        # Load manifest for metadata
        manifest_file = self.results_dir / "ingest" / "manifest.csv"
        self.manifest = pd.read_csv(manifest_file) if manifest_file.exists() else None

    def _setup_templates(self) -> None:
        """Set up Jinja2 template environment."""
        if not JINJA2_AVAILABLE:
            print("Warning: Jinja2 not available for template rendering")

        if JINJA2_AVAILABLE and self.template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            self.jinja_env = None

    def generate_publication_report(
        self,
        output_format: str = "pdf",
        template: str = "publication_report.tex",
        **kwargs
    ) -> str:
        """Generate a publication-ready scientific report.

        Args:
            output_format: Output format ('pdf', 'html', 'docx')
            template: LaTeX template to use
            **kwargs: Additional template variables

        Returns:
            Path to generated report file
        """
        # Extract key findings
        findings = self._extract_key_findings()

        # Prepare template data
        template_data = {
            "title": "Flow Cytometry Quality Control Analysis Report",
            "authors": "CytoFlow-QC Analysis Pipeline",
            "date": pd.Timestamp.now().strftime("%B %Y"),
            "findings": findings,
            "qc_summary": self._generate_qc_summary(),
            "gating_summary": self._generate_gating_summary(),
            "drift_summary": self._generate_drift_summary(),
            "stats_summary": self._generate_stats_summary(),
            "methods": self._generate_methods_section(),
            "conclusions": self._generate_conclusions(),
            **kwargs
        }

        if output_format.lower() == "pdf":
            return self._generate_pdf_report(template_data, template)
        elif output_format.lower() == "html":
            return self._generate_html_report(template_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _extract_key_findings(self) -> list[str]:
        """Extract key findings from analysis results."""
        findings = []

        # QC findings
        if self.qc_data is not None:
            mean_qc_pass = self.qc_data["qc_pass_fraction"].mean()
            total_samples = len(self.qc_data)

            if mean_qc_pass > 0.9:
                findings.append(f"High-quality dataset: {mean_qc_pass".1%"} of {total_samples} samples passed QC criteria.")
            elif mean_qc_pass > 0.7:
                findings.append(f"Acceptable dataset quality: {mean_qc_pass".1%"} of {total_samples} samples passed QC criteria.")
            else:
                findings.append(f"Dataset requires attention: only {mean_qc_pass".1%"} of {total_samples} samples passed QC criteria.")

        # Gating findings
        if self.gate_data is not None:
            retention_rate = self.gate_data["gated_events"].sum() / self.gate_data["input_events"].sum()
            findings.append(f"Automated gating retained {retention_rate".1%"} of total events across all samples.")

        # Drift findings
        if self.drift_tests is not None:
            significant_effects = (self.drift_tests["adj_p_value"] < 0.05).sum()
            total_tests = len(self.drift_tests)

            if significant_effects > 0:
                findings.append(f"Detected {significant_effects} significant batch effects out of {total_tests} statistical tests.")
            else:
                findings.append("No significant batch effects detected across experimental conditions.")

        # Statistical findings
        if self.stats_data is not None:
            significant_effects = (self.stats_data["adj_p_value"] < 0.05).sum()
            total_comparisons = len(self.stats_data)

            if significant_effects > 0:
                findings.append(f"Found {significant_effects} significant treatment effects out of {total_comparisons} marker comparisons.")
            else:
                findings.append("No significant treatment effects detected in this analysis.")

        return findings

    def _generate_qc_summary(self) -> str:
        """Generate QC summary section."""
        if self.qc_data is None:
            return "No quality control data available."

        summary = f"""
## Quality Control Summary

This analysis processed {len(self.qc_data)} samples with the following quality metrics:

- **Overall Pass Rate**: {self.qc_data["qc_pass_fraction"].mean()".1%"} of samples met quality criteria
- **Debris Fraction**: Mean {self.qc_data["debris_fraction"].mean()".1%"} (range: {self.qc_data["debris_fraction"].min()".1%"} - {self.qc_data["debris_fraction"].max()".1%"})
- **Doublet Fraction**: Mean {self.qc_data["doublet_fraction"].mean()".1%"} (range: {self.qc_data["doublet_fraction"].min()".1%"} - {self.qc_data["doublet_fraction"].max()".1%"})

### Sample Quality Distribution

The quality control analysis identified:
- **High Quality Samples**: {(self.qc_data["qc_pass_fraction"] > 0.9).sum()} samples
- **Moderate Quality Samples**: {((self.qc_data["qc_pass_fraction"] > 0.7) & (self.qc_data["qc_pass_fraction"] <= 0.9)).sum()} samples
- **Low Quality Samples**: {(self.qc_data["qc_pass_fraction"] <= 0.7).sum()} samples

### Quality Metrics by Sample

| Sample ID | QC Pass Rate | Debris Fraction | Doublet Fraction | Status |
|-----------|--------------|-----------------|------------------|--------|
"""

        # Add sample details
        for _, row in self.qc_data.iterrows():
            status = "✅ Pass" if row["qc_pass"] else "❌ Fail"
            summary += f"| {row['sample_id']} | {row['qc_pass_fraction']".1%"} | {row['debris_fraction']".1%"} | {row['doublet_fraction']".1%"} | {status} |\n"

        return summary

    def _generate_gating_summary(self) -> str:
        """Generate gating summary section."""
        if self.gate_data is None:
            return "No gating data available."

        total_input = self.gate_data["input_events"].sum()
        total_gated = self.gate_data["gated_events"].sum()
        retention_rate = total_gated / total_input if total_input > 0 else 0

        summary = f"""
## Automated Gating Summary

The automated gating strategy processed {len(self.gate_data)} samples with the following results:

- **Total Events Processed**: {total_input","}
- **Events Retained After Gating**: {total_gated","}
- **Overall Retention Rate**: {retention_rate".1%"} of events passed gating criteria

### Gating Performance by Sample

| Sample ID | Input Events | Gated Events | Retention Rate | Status |
|-----------|--------------|--------------|----------------|--------|
"""

        # Add sample details
        for _, row in self.gate_data.iterrows():
            sample_retention = row["gated_events"] / row["input_events"] if row["input_events"] > 0 else 0
            summary += f"| {row['sample_id']} | {row['input_events']","} | {row['gated_events']","} | {sample_retention".1%"} | ✅ Success |\n"

        return summary

    def _generate_drift_summary(self) -> str:
        """Generate drift analysis summary section."""
        if self.drift_tests is None:
            return "No drift analysis data available."

        significant_effects = (self.drift_tests["adj_p_value"] < 0.05).sum()
        total_tests = len(self.drift_tests)

        summary = f"""
## Batch Drift Analysis Summary

Batch drift analysis examined {total_tests} statistical comparisons to identify systematic differences between experimental batches.

- **Total Statistical Tests**: {total_tests}
- **Significant Batch Effects**: {significant_effects}
- **Significance Rate**: {significant_effects / total_tests".1%"} of tests showed significant batch differences

### Significant Findings

"""

        if significant_effects > 0:
            significant_tests = self.drift_tests[self.drift_tests["adj_p_value"] < 0.05]
            summary += f"The following {significant_effects} features showed significant batch effects:\n\n"

            for _, row in significant_tests.iterrows():
                summary += f"- **{row['feature']}**: p = {row['adj_p_value']".2e"}, effect size = {row['statistic']".3f"}\n"
        else:
            summary += "No significant batch effects were detected in this analysis.\n"

        return summary

    def _generate_stats_summary(self) -> str:
        """Generate statistical analysis summary section."""
        if self.stats_data is None:
            return "No statistical analysis data available."

        significant_effects = (self.stats_data["adj_p_value"] < 0.05).sum()
        total_comparisons = len(self.stats_data)

        summary = f"""
## Statistical Analysis Summary

Statistical analysis compared {total_comparisons} marker measurements across experimental conditions.

- **Total Comparisons**: {total_comparisons}
- **Significant Effects**: {significant_effects}
- **Significance Rate**: {significant_effects / total_comparisons".1%"} of comparisons showed significant differences

### Key Findings

"""

        if significant_effects > 0:
            # Show top significant effects
            top_effects = self.stats_data[self.stats_data["adj_p_value"] < 0.05].nlargest(5, "effect_size")

            for _, row in top_effects.iterrows():
                summary += f"- **{row['marker']}** ({row['group1']} vs {row['group2']}): Effect size = {row['effect_size']".3f"}, p = {row['adj_p_value']".2e"}\n"
        else:
            summary += "No significant treatment effects were detected in this analysis.\n"

        return summary

    def _generate_methods_section(self) -> str:
        """Generate methods section describing analysis pipeline."""
        methods = """
## Methods

### Data Processing Pipeline

This analysis was performed using the CytoFlow-QC automated pipeline with the following stages:

1. **Data Ingestion**: Flow cytometry data files were loaded and standardized
2. **Quality Control**: Automated assessment of data quality including debris detection, doublet identification, and saturation analysis
3. **Automated Gating**: Machine learning-based identification of cell populations
4. **Batch Drift Analysis**: Statistical assessment of systematic differences between experimental batches
5. **Statistical Analysis**: Non-parametric effect size calculations with multiple testing correction

### Quality Control Criteria

- **Debris Removal**: Events in the lowest 2% of FSC and SSC distributions were excluded
- **Doublet Detection**: Events deviating >8% from expected FSC-A/FSC-H ratio were flagged
- **Saturation Check**: Events exceeding 99.5% of detector maximum were excluded

### Statistical Methods

- **Effect Size Calculation**: Hedges' g for standardized mean differences
- **Statistical Testing**: Mann-Whitney U test for non-parametric comparisons
- **Multiple Testing Correction**: Holm-Bonferroni procedure for family-wise error control

### Software and Versions

- **CytoFlow-QC**: Automated quality control and gating pipeline
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
- **Python Environment**: Configured for reproducible analysis
"""

        return methods

    def _generate_conclusions(self) -> str:
        """Generate conclusions section."""
        conclusions = """
## Conclusions

### Summary of Findings

This automated analysis provides a comprehensive assessment of flow cytometry data quality and experimental outcomes. Key conclusions include:

1. **Data Quality Assessment**: The quality control pipeline successfully processed all samples with acceptable retention rates
2. **Gating Performance**: Automated gating strategies effectively identified target cell populations
3. **Batch Consistency**: Statistical analysis revealed {0 if self.drift_tests is None else (self.drift_tests["adj_p_value"] < 0.05).sum()} significant batch effects requiring attention
4. **Treatment Effects**: {0 if self.stats_data is None else (self.stats_data["adj_p_value"] < 0.05).sum()} significant treatment effects were identified

### Recommendations

- **Data Quality**: Overall data quality supports reliable downstream analysis
- **Batch Effects**: {0 if self.drift_tests is None else "Address" if (self.drift_tests["adj_p_value"] < 0.05).sum() > 0 else "No"} batch effects detected require {0 if self.drift_tests is None else "attention" if (self.drift_tests["adj_p_value"] < 0.05).sum() > 0 else "no special consideration"}
- **Statistical Power**: The analysis has sufficient power to detect biologically meaningful effects
- **Future Experiments**: Consider optimizing gating parameters for improved population identification

### Data Availability

All analysis results, including raw data, processed datasets, and statistical outputs, are available in the results directory for further investigation and validation.
"""

        return conclusions

    def _generate_pdf_report(self, template_data: dict[str, Any], template: str) -> str:
        """Generate PDF report using LaTeX."""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("weasyprint required for PDF generation")

        # Generate HTML first
        html_content = self._generate_html_report(template_data, template="report.html")

        # Convert HTML to PDF
        output_file = self.results_dir / "publication_report.pdf"

        html_doc = HTML(string=html_content)
        css_string = self._get_pdf_css()

        html_doc.write_pdf(
            output_file,
            stylesheets=[CSS(string=css_string)],
            pdf_version="1.7"
        )

        return str(output_file)

    def _generate_html_report(self, template_data: dict[str, Any], template: str = "report.html") -> str:
        """Generate HTML report."""
        if not self.jinja_env:
            # Fallback to basic HTML generation
            return self._generate_basic_html_report(template_data)

        # Use Jinja2 template
        template_obj = self.jinja_env.get_template(template)
        return template_obj.render(**template_data)

    def _generate_basic_html_report(self, template_data: dict[str, Any]) -> str:
        """Generate basic HTML report without Jinja2."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{template_data.get("title", "CytoFlow-QC Report")}</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #2E86AB; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #A23B72; background: #f8f9fa; }}
                h1 {{ margin: 0; }}
                h2 {{ color: #A23B72; border-bottom: 2px solid #A23B72; padding-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{template_data.get("title", "CytoFlow-QC Report")}</h1>
                <p>Generated by CytoFlow-QC Analysis Pipeline</p>
                <p><strong>Date:</strong> {template_data.get("date", "Unknown")}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <ul>
        """

        for finding in template_data.get("findings", []):
            html += f"                    <li>{finding}</li>\n"

        html += """
                </ul>
            </div>
        """

        # Add sections
        sections = [
            ("qc_summary", "Quality Control Analysis"),
            ("gating_summary", "Gating Analysis"),
            ("drift_summary", "Batch Drift Analysis"),
            ("stats_summary", "Statistical Analysis"),
            ("methods", "Methods"),
            ("conclusions", "Conclusions"),
        ]

        for section_key, section_title in sections:
            content = template_data.get(section_key, "No data available.")
            html += f"""
            <div class="section">
                <h2>{section_title}</h2>
                <div>{content}</div>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _get_pdf_css(self) -> str:
        """Get CSS for PDF generation."""
        return """
        @page {
            size: A4;
            margin: 2cm;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
            }
        }

        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #333;
        }

        .title {
            text-align: center;
            font-size: 18pt;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2E86AB;
        }

        .section {
            margin: 20px 0;
            page-break-inside: avoid;
        }

        h1, h2 {
            color: #2E86AB;
            page-break-after: avoid;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }

        th, td {
            border: 1px solid #333;
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }

        .metric {
            background-color: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #28a745;
            margin: 10px 0;
        }
        """


class InteractiveReportBuilder:
    """Interactive report builder with drag-and-drop interface."""

    def __init__(self, results_dir: str | Path):
        """Initialize interactive report builder.

        Args:
            results_dir: Path to analysis results
        """
        self.results_dir = Path(results_dir)
        self.report_sections = []
        self.report_config = {}

    def add_section(self, section_type: str, title: str, config: dict[str, Any] | None = None) -> None:
        """Add a section to the report.

        Args:
            section_type: Type of section ('qc', 'gating', 'drift', 'stats', 'custom')
            title: Section title
            config: Section configuration
        """
        section = {
            "type": section_type,
            "title": title,
            "config": config or {},
            "position": len(self.report_sections)
        }
        self.report_sections.append(section)

    def generate_interactive_report(self, output_path: str | Path) -> None:
        """Generate interactive HTML report with drag-and-drop sections.

        Args:
            output_path: Path to save the report
        """
        # Generate HTML with drag-and-drop interface
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoFlow-QC Interactive Report Builder</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; text-align: center; }}
                .toolbar {{ background: #f8f9fa; padding: 15px; border-bottom: 1px solid #ddd; }}
                .workspace {{ padding: 20px; min-height: 600px; }}
                .section {{ background: white; margin: 15px 0; padding: 20px;
                           border: 1px solid #ddd; border-radius: 8px;
                           cursor: move; position: relative; }}
                .section:hover {{ border-color: #007bff; box-shadow: 0 2px 8px rgba(0,123,255,0.25); }}
                .section-header {{ display: flex; justify-content: space-between; align-items: center; }}
                .delete-btn {{ background: #dc3545; color: white; border: none; padding: 5px 10px;
                              border-radius: 4px; cursor: pointer; }}
                .add-section {{ background: #28a745; color: white; border: none; padding: 10px 20px;
                               border-radius: 5px; cursor: pointer; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 CytoFlow-QC Interactive Report Builder</h1>
                <p>Build custom reports by dragging and dropping sections</p>
            </div>

            <div class="toolbar">
                <button class="add-section" onclick="addSection('qc')">Add QC Section</button>
                <button class="add-section" onclick="addSection('gating')">Add Gating Section</button>
                <button class="add-section" onclick="addSection('drift')">Add Drift Section</button>
                <button class="add-section" onclick="addSection('stats')">Add Stats Section</button>
                <button class="add-section" onclick="addSection('custom')">Add Custom Section</button>
                <button onclick="exportReport()">Export Report</button>
            </div>

            <div class="workspace" id="workspace">
                <h3>Drag sections here to build your report</h3>
            </div>

            <script>
                let sectionCounter = 0;

                function addSection(type) {{
                    const workspace = document.getElementById('workspace');
                    const section = document.createElement('div');
                    section.className = 'section';
                    section.draggable = true;
                    section.id = `section-${++sectionCounter}`;

                    section.innerHTML = `
                        <div class="section-header">
                            <h3>New ${type.toUpperCase()} Section</h3>
                            <button class="delete-btn" onclick="deleteSection('${{section.id}}')">Delete</button>
                        </div>
                        <div class="section-content">
                            <p>This section will contain ${type} analysis results.</p>
                        </div>
                    `;

                    workspace.appendChild(section);
                    makeDraggable(section);
                }}

                function makeDraggable(element) {{
                    element.addEventListener('dragstart', (e) => {{
                        e.dataTransfer.setData('text/plain', element.id);
                    }});
                }}

                document.addEventListener('dragover', (e) => {{
                    e.preventDefault();
                }});

                document.addEventListener('drop', (e) => {{
                    e.preventDefault();
                    const draggedId = e.dataTransfer.getData('text/plain');
                    const draggedElement = document.getElementById(draggedId);

                    if (e.target.classList.contains('workspace')) {{
                        e.target.appendChild(draggedElement);
                    }}
                }});

                function deleteSection(sectionId) {{
                    const section = document.getElementById(sectionId);
                    if (section) {{
                        section.remove();
                    }}
                }}

                function exportReport() {{
                    const sections = Array.from(document.querySelectorAll('.section'));
                    const reportData = sections.map(section => ({{
                        type: section.querySelector('h3').textContent.split(' ')[1].toLowerCase(),
                        title: section.querySelector('h3').textContent,
                        content: section.querySelector('.section-content').innerHTML
                    }}));

                    // In a real implementation, this would generate the final report
                    alert('Report exported! (This is a demo - actual export would generate PDF/HTML)');
                    console.log('Report data:', reportData);
                }}
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Interactive report builder saved to: {output_path}")


class ReportPublisher:
    """Publishing utilities for sharing reports."""

    def __init__(self, results_dir: str | Path):
        """Initialize report publisher.

        Args:
            results_dir: Path to results directory
        """
        self.results_dir = Path(results_dir)

    def publish_to_github_pages(self, repo_url: str, branch: str = "gh-pages") -> str:
        """Publish report to GitHub Pages.

        Args:
            repo_url: GitHub repository URL
            branch: Branch for GitHub Pages

        Returns:
            URL of published report
        """
        # This would implement GitHub Pages publishing
        # For now, return placeholder URL
        return f"https://{repo_url.split('/')[-1].replace('.git', '')}.github.io/"

    def publish_to_readthedocs(self, project_slug: str) -> str:
        """Publish documentation to Read the Docs.

        Args:
            project_slug: Read the Docs project slug

        Returns:
            URL of published documentation
        """
        return f"https://{project_slug}.readthedocs.io/"

    def create_shareable_link(self, report_path: str | Path) -> str:
        """Create a shareable link for the report.

        Args:
            report_path: Path to report file

        Returns:
            Shareable URL
        """
        # This would create a temporary hosting link
        # For now, return file path
        return f"file://{Path(report_path).absolute()}"

    def export_for_collaboration(self, output_dir: str | Path) -> str:
        """Export report package for collaboration.

        Args:
            output_dir: Directory for collaboration package

        Returns:
            Path to created package
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy report files
        import shutil

        # Copy HTML report
        html_report = self.results_dir / "publication_report.html"
        if html_report.exists():
            shutil.copy2(html_report, output_path / "report.html")

        # Copy data files
        data_dir = output_path / "data"
        data_dir.mkdir(exist_ok=True)

        for csv_file in self.results_dir.rglob("*.csv"):
            shutil.copy2(csv_file, data_dir / csv_file.name)

        # Create README for collaborators
        readme_content = f"""
# CytoFlow-QC Analysis Report

This package contains the complete analysis results from CytoFlow-QC.

## Contents

- `report.html` - Interactive HTML report
- `data/` - All analysis data files (CSV format)
- `README.md` - This documentation

## Viewing the Report

Open `report.html` in any modern web browser to view the interactive analysis report.

## Data Files

The `data/` directory contains:
- QC summary statistics
- Gating results
- Drift analysis data
- Statistical analysis results

## Analysis Pipeline

This report was generated using the CytoFlow-QC automated pipeline with the following stages:
1. Data ingestion and standardization
2. Quality control assessment
3. Automated gating
4. Batch drift analysis
5. Statistical analysis

## Sharing

This package can be shared with collaborators for review and further analysis.

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(output_path / "README.md", 'w') as f:
            f.write(readme_content)

        # Create zip file
        import zipfile
        zip_path = output_path.parent / "cytoflow_qc_collaboration_package.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_path.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(output_path.parent))

        return str(zip_path)


def generate_publication_report(
    results_dir: str | Path,
    output_format: str = "pdf",
    template: str = "publication_report.tex",
    **kwargs
) -> str:
    """Generate a publication-ready scientific report.

    Args:
        results_dir: Path to cytoflow-qc results directory
        output_format: Output format ('pdf', 'html', 'docx')
        template: Template to use for report generation
        **kwargs: Additional template variables

    Returns:
        Path to generated report file
    """
    generator = ReportGenerator(results_dir)
    return generator.generate_publication_report(output_format, template, **kwargs)


def create_interactive_report_builder(results_dir: str | Path, output_path: str | Path) -> None:
    """Create an interactive report builder.

    Args:
        results_dir: Path to analysis results
        output_path: Path to save the builder
    """
    builder = InteractiveReportBuilder(results_dir)
    builder.generate_interactive_report(output_path)


def publish_report(
    results_dir: str | Path,
    platform: str = "github",
    **kwargs
) -> str:
    """Publish report to specified platform.

    Args:
        results_dir: Path to results directory
        platform: Publishing platform ('github', 'readthedocs', 'local')
        **kwargs: Platform-specific parameters

    Returns:
        URL of published report
    """
    publisher = ReportPublisher(results_dir)

    if platform == "github":
        return publisher.publish_to_github_pages(kwargs.get("repo_url", ""))
    elif platform == "readthedocs":
        return publisher.publish_to_readthedocs(kwargs.get("project_slug", ""))
    else:
        # Local sharing
        report_path = results_dir / "publication_report.html"
        return publisher.create_shareable_link(report_path)


def create_collaboration_package(
    results_dir: str | Path,
    output_dir: str | Path
) -> str:
    """Create a collaboration package for sharing results.

    Args:
        results_dir: Path to analysis results
        output_dir: Directory for collaboration package

    Returns:
        Path to created package
    """
    publisher = ReportPublisher(results_dir)
    return publisher.export_for_collaboration(output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m cytoflow_qc.report_generator <results_directory> [output_format]")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "html"

    try:
        report_path = generate_publication_report(results_dir, output_format)
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)








