from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_env(prompts_dir: str):
    loader = FileSystemLoader(prompts_dir)
    env = Environment(
        loader=loader,
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def render_prompt(env: Environment, template_name: str, **kwargs) -> str:
    template = env.get_template(template_name)
    return template.render(**kwargs)
