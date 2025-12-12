"""Interactive configuration generator for CytoFlow-QC."""
import questionary
import yaml
from pathlib import Path

from cytoflow_qc.config import AppConfig

def generate_config_interactive():
    """Launch an interactive questionnaire to generate a config.yaml file."""
    print("Welcome to the CytoFlow-QC interactive configuration generator!")
    print("This will guide you through creating a `config.yaml` file.")
    print("Press Enter to accept the default value in brackets.")

    defaults = AppConfig().model_dump()
    config = {}

    # --- Channels ---
    questionary.print("\n--- Channel Configuration ---", style="bold underline")
    channels = {}
    channels['fsc_a'] = questionary.text("Forward scatter channel (FSC-A):", default=defaults['channels']['fsc_a']).ask()
    channels['fsc_h'] = questionary.text("Forward scatter height channel (FSC-H):", default=defaults['channels']['fsc_h']).ask()
    channels['ssc_a'] = questionary.text("Side scatter channel (SSC-A):", default=defaults['channels']['ssc_a']).ask()
    if questionary.confirm("Do you have a viability channel (e.g., Zombie Dye)?", default=bool(defaults['channels']['viability'])).ask():
        channels['viability'] = questionary.text("Viability channel name:", default=defaults['channels']['viability'] or 'Zombie-A').ask()
    markers_str = questionary.text("Enter your marker channels (comma-separated):", default=", ".join(defaults['channels']['markers'])).ask()
    channels['markers'] = [m.strip() for m in markers_str.split(',') if m.strip()]
    config['channels'] = channels

    # --- QC ---
    questionary.print("\n--- Quality Control (QC) Configuration ---", style="bold underline")
    qc = {}
    qc['debris'] = {
        'method': 'percentile',
        'fsc_a_pct': int(questionary.text("Debris FSC percentile threshold:", default=str(defaults['qc']['debris']['fsc_a_pct'])).ask()),
        'ssc_a_pct': int(questionary.text("Debris SSC percentile threshold:", default=str(defaults['qc']['debris']['ssc_a_pct'])).ask()),
    }
    config['qc'] = qc
    
    # --- Gating ---
    questionary.print("\n--- Gating Configuration ---", style="bold underline")
    gating = {}
    gating['lymphocytes'] = {
        'method': 'density',
        'percentile': int(questionary.text("Lymphocyte gate density percentile:", default=str(defaults['gating']['lymphocytes']['percentile'])).ask()),
    }
    config['gating'] = gating

    # --- Save File ---
    questionary.print("\n--- Saving Configuration ---", style="bold underline")
    save_path_str = questionary.text("Path to save config file:", default="config.yaml").ask()
    save_path = Path(save_path_str)

    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        print(f"\n✅ Configuration saved successfully to {save_path}")
    except IOError as e:
        print(f"\n❌ Error saving configuration file: {e}")

if __name__ == '__main__':
    generate_config_interactive()
