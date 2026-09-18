if [[ "$VIRTUAL_ENV" == "" ]]; then
    python3 -m venv .venv
    source .venv/bin/activate
    if [[ -e .env ]]; then
      export $(cat .env | xargs)
    fi
fi