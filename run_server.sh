#!/usr/bin/env bash
# Starts the translation server in its own virtualenv. Extra args go to the server.
cd "$(dirname "$0")"
. .venv/bin/activate
exec python -m translation_server.server "$@"
