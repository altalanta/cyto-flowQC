"""Plugin scaffolding tool for CytoFlow-QC."""
from pathlib import Path
from cookiecutter.main import cookiecutter
import questionary

def create_plugin_scaffold():
    """Interactively generate a new plugin from the template."""
    print("Welcome to the CytoFlow-QC plugin generator!")
    
    template_path = Path(__file__).parent / "templates" / "cookiecutter-cytoflow-qc-plugin"
    
    # --- Collect User Input ---
    plugin_name = questionary.text(
        "What is the name of your new plugin?",
        default="My Custom Gating"
    ).ask()

    author_name = questionary.text(
        "What is your name?",
        default="Plugin Developer"
    ).ask()

    author_email = questionary.text(
        "What is your email?",
        default="developer@example.com"
    ).ask()

    if not all([plugin_name, author_name, author_email]):
        print("Plugin generation cancelled.")
        return

    # --- Run Cookiecutter ---
    try:
        cookiecutter(
            str(template_path),
            no_input=True,
            extra_context={
                "plugin_name": plugin_name,
                "author_name": author_name,
                "author_email": author_email,
            }
        )
        project_slug = plugin_name.lower().replace(' ', '_').replace('-', '_')
        print(f"\n✅ Plugin '{plugin_name}' created in ./{project_slug}/")
        print("To get started, run:")
        print(f"  cd {project_slug}")
        print("  pip install -e .")
    except Exception as e:
        print(f"\n❌ Error generating plugin: {e}")

if __name__ == '__main__':
    create_plugin_scaffold()
