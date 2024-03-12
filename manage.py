#!/usr/bin/env python
import os
import sys
import json


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    with open(os.path.join(os.path.dirname(__file__), "config/secrets.json"), "r") as f:
        os.environ["OPENAI_API_KEY"] = json.load(f)["OPENAI_API_KEY"]

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
